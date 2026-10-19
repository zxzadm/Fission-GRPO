import math
import re
import json
import os
from collections import Counter
from typing import List
from transformers import AutoTokenizer

model_name = os.environ.get("BASE_MODEL")
print("_____model_name_____" * 10, model_name)

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None and model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer

# 你的总训练步数：3 * 337
MAX_DYNAMIC_STEP = 100

# ==================== 1. format reward ====================
def format_reward(data_source, solution_str, ground_truth, extra_info):
    step = extra_info.get("step", 10000)
    if step is None:
        print("step检查为none，后续需要进一步排查")
        step = 0
    step = max(0, min(step, MAX_DYNAMIC_STEP))
    ratio = step / MAX_DYNAMIC_STEP if MAX_DYNAMIC_STEP > 0 else 1.0
    print("当前format_step:", step)

    # Adjusted max: 3 -> 2, min: 0 -> 0 (range [0, 2] with step-based variation)
    base_max = 3.0 - ratio  # From 3 -> 2 as step increases
    base_min = 0.0
    max_possible_reward = base_max # max_possible_reward -> [0, 2]
    min_possible_reward = base_min  # min_possible_reward is always 0

    pattern = r"^<think>\n.*?\n</think>\n\n<tool_calls>\n.*?\n</tool_calls>$"

    if isinstance(solution_str, str):
        completions = [solution_str]
    else:
        completions = solution_str

    rewards = []
    for content in completions:
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        rewards.append(max_possible_reward if match else min_possible_reward)

    print("Format rewards:", rewards)
    return rewards

# ==================== 2. acc reward ====================
def match_score(list1, list2):
    if list1 == list2:
        return 1.0
    if not list1 or not list2:
        return 0.0
    c1, c2 = Counter(list1), Counter(list2)
    inter = sum(min(c1[k], c2[k]) for k in c1.keys() & c2.keys())
    max_possible = len(list1) + len(list2) - inter
    return inter / max_possible if max_possible > 0 else 0.0

def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    if gt_tools == pd_tools:
        return max_possible_reward

    gt_names = [t["name"] for t in gt_tools]
    pd_names = [t["name"] for t in pd_tools]
    score = match_score(gt_names, pd_names)

    local_max_possible = 1.0
    used = set()

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["arguments"]
        local_max_possible += 1.0 + 2.0

        best_score = 0.0
        best_idx = -1

        for i, pd_tool in enumerate(pd_tools):
            if i in used or pd_tool["name"] != gt_name:
                continue

            pd_params = pd_tool["arguments"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            correctness = sum(
                1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v
            )
            tp = correctness
            fp = len(pd_params) - correctness
            fn = len(gt_params) - correctness
            acc = tp / (tp + fp + 1e-30)
            recall = tp / (tp + fn + 1e-30)
            if acc + recall > 0:
                f1 = 2 * acc * recall / (acc + recall)
            else:
                f1 = 0.0

            total = param_score + 2 * f1
            if total > best_score:
                best_score = total
                best_idx = i

        if best_idx != -1:
            used.add(best_idx)
            score += best_score

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

def acc_reward(data_source, solution_str, ground_truth, extra_info):
    step = extra_info.get("step", 10000)
    if step is None:
        print("step检查为none，后续需要进一步排查")
        step = 0
    step = max(0, min(step, MAX_DYNAMIC_STEP))
    ratio = step / MAX_DYNAMIC_STEP if MAX_DYNAMIC_STEP > 0 else 1.0
    print("当前acc_step:", step)

    # Adjusted max: 2 -> 3, min: 0 -> 0 (range [0, 3] with step-based variation)
    base_max = 2.0 + ratio  # From 2 -> 3 as step increases
    base_min = 0.0
    max_possible_reward = base_max # max_possible_reward -> [0, 3]
    min_possible_reward = base_min  # min_possible_reward is always 0

    if isinstance(solution_str, str):
        responses = [solution_str]
    else:
        responses = solution_str
    if isinstance(ground_truth, str):
        answers = [ground_truth]
    else:
        answers = ground_truth

    assert len(responses) == len(answers)

    rewards = []
    for resp, ans in zip(responses, answers):
        try:
            gt_tools = json.loads(ans.strip())
            assert isinstance(gt_tools, list)
            assert "<tool_calls>" in resp and "</tool_calls>" in resp
            pd_str = resp.split("<tool_calls>")[1].split("</tool_calls>")[0].strip()
            pd_tools = json.loads(pd_str)
            assert isinstance(pd_tools, list)

            r = compute_tool_call_reward(
                gt_tools, pd_tools,
                max_possible_reward, min_possible_reward
            )
        except Exception:
            r = min_possible_reward
        rewards.append(r)

    print("Final accuracy rewards:", rewards)
    return rewards

# ==================== 3. len reward ====================
def extract_think(text):
    """提取<think>标签中的内容，找不到标签返回空字符串"""
    m = re.search(r"<think>\n(.*?)\n</think>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 找不到标签时返回空字符串，这样会触发极短文本惩罚(reward=0)
    return ""

# ==================== 新版 len_reward ====================
def len_reward(data_source, solution_str, ground_truth, extra_info, tokenizer=None):
    """
    长度奖励函数 (固定参数版本):
    - 极短文本: < 64 tokens → reward = 0
    - 短文本惩罚: 64-128 tokens → 线性增长 (0 → 1.0)
    - 理想区间: 128-1024 tokens → reward = 1.0 (满分)
    - 长文本惩罚: 1024-2048 tokens → 高斯衰减 (1.0 → 0)
    - 极长文本: >= 2048 tokens → reward = 0
    """
    if tokenizer is None:
        tok = get_tokenizer()
        if tok is None:
            raise ValueError("Tokenizer未初始化，请设置BASE_MODEL环境变量")
    else:
        tok = tokenizer

    # ===== 关键参数定义 (固定值) =====
    ULTRA_SHORT_CUTOFF = 16   # 极短文本硬截断
    SHORT_PENALTY_START = 16   # 短文本惩罚起点
    IDEAL_MIN = 64          # 理想最小长度
    IDEAL_MAX = 1024           # 理想最大长度
    LONG_PENALTY_START = 1024  # 长文本惩罚起点
    ULTRA_LONG_CUTOFF =1920    # 极长文本硬截断
    SIGMA = 250                # 长文本惩罚的sigma值 (固定)

    if isinstance(solution_str, str):
        completions = [solution_str]
    else:
        completions = solution_str

    rewards = []
    for content in completions:
        think = extract_think(content)
        
        # 如果think为空（找不到标签），长度为0，会触发极短文本惩罚
        gen_len = len(tok.encode(think))
        
        if gen_len == 0:
            print(f"Token length: {gen_len} (格式错误: 缺少<think>标签或标签为空)")
        else:
            print(f"Token length: {gen_len}")

        # ===== 1. 极短文本: < 64 =====
        if gen_len < ULTRA_SHORT_CUTOFF:
            reward = 0.0
            print(f"  -> 极短文本 (< {ULTRA_SHORT_CUTOFF}): reward = {reward:.4f}")
        
        # ===== 2. 短文本惩罚区间: [64, 128) =====
        elif SHORT_PENALTY_START <= gen_len < IDEAL_MIN:
            # 线性增长从 0 到 1.0
            normalized = (gen_len - SHORT_PENALTY_START) / (IDEAL_MIN - SHORT_PENALTY_START)
            reward = normalized
            print(f"  -> 短文本惩罚区间 [{SHORT_PENALTY_START}, {IDEAL_MIN}): normalized={normalized:.3f}, reward={reward:.4f}")
        
        # ===== 3. 理想区间: [128, 1024] =====
        elif IDEAL_MIN <= gen_len <= IDEAL_MAX:
            reward = 1.0
            print(f"  -> 理想区间 [{IDEAL_MIN}, {IDEAL_MAX}]: reward = {reward:.4f}")
        
        # ===== 4. 长文本惩罚区间: (1024, 2048) =====
        elif LONG_PENALTY_START < gen_len < ULTRA_LONG_CUTOFF:
            # 使用高斯衰减
            excess = gen_len - LONG_PENALTY_START
            reward = math.exp(-(excess ** 2) / (2 * SIGMA ** 2))
            print(f"  -> 长文本惩罚区间 ({LONG_PENALTY_START}, {ULTRA_LONG_CUTOFF}): excess={excess}, sigma={SIGMA}, reward={reward:.4f}")
        
        # ===== 5. 极长文本: >= 2048 =====
        else:
            reward = 0.0
            print(f"  -> 极长文本 (>= {ULTRA_LONG_CUTOFF}): reward = {reward:.4f}")

        rewards.append(reward)

    print(f"Final length rewards: {rewards}")
    return rewards

# ==================== 4. 主接口 ====================
def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    外面只需要把 DataProto 当 data_source 传进来，不需要显式写 step
    """
    res_f = format_reward(data_source, solution_str, ground_truth, extra_info)
    res_a = acc_reward(data_source, solution_str, ground_truth, extra_info)
    res_l = len_reward(data_source, solution_str, ground_truth, extra_info)

    total = (res_f[0] + res_a[0] + res_l[0]) / 3.0

    return {
        "score": total,
        "format_score": res_f[0],
        "acc_score": res_a[0],
        "len_score": res_l[0],
    }