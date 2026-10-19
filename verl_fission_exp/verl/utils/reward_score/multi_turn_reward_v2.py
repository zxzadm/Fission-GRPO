import sys
import math
import re
import json
import os
import string
from collections import Counter
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer

model_name = os.environ.get("BASE_MODEL")
print("_____model_name_____" * 10, model_name)

# 全局tokenizer实例
_tokenizer = None

def get_tokenizer():
    """懒加载tokenizer"""
    global _tokenizer
    if _tokenizer is None and model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


# ============= Format Reward =============
def format_reward(data_source, solution_str, ground_truth, extra_info=None):
    """检查completion是否符合特定格式"""
    # 固定奖励值（不随step变化）
    max_possible_reward = 2.0
    min_possible_reward = 0.0
    
    # 格式匹配pattern
    pattern = r"^<think>\n.*?\n</think>\n\n<tool_calls>\n.*?\n</tool_calls>$"
    
    # 处理输入：统一转换为列表
    if isinstance(solution_str, str):
        completions = [solution_str]
    else:
        completions = solution_str
    
    # 匹配格式
    rewards = []
    for content in completions:
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        reward = max_possible_reward if match else min_possible_reward
        rewards.append(reward)
    
    print('Format rewards:', rewards)
    
    return rewards


# ============= Accuracy Reward =============
def match_score(list1, list2):
    """计算两个列表的相似度分数(考虑元素频率,忽略顺序)"""
    if list1 == list2:
        return 1.0
    
    
    if not list1 or not list2:
        return 0.0
    
    count1 = Counter(list1)
    count2 = Counter(list2)
    
    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection
    
    return intersection / max_possible if max_possible > 0 else 0.0


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward):
    """计算tool call的准确性奖励"""
    if gt_tools == pd_tools:
        return max_possible_reward
    
    # 提取工具名称
    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(gt_names, pd_names)
    
    local_max_possible = 1.0
    used_pd_indices = set()
    
    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["arguments"]
        
        local_max_possible += 1.0 + 2.0
        
        best_match_score = 0.0
        best_match_index = -1
        
        # 找到最佳匹配的pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue
            
            pd_params = pd_tool["arguments"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))
            
            # 计算参数值的正确性
            correctness_score = sum(1.0 for k, v in gt_params.items() 
                                   if k in pd_params and pd_params[k] == v)
            tp = correctness_score
            fp = len(pd_params) - correctness_score
            fn = len(gt_params) - correctness_score
            
            acc = tp / (tp + fp + 1e-30)
            recall = tp / (tp + fn + 1e-30)
            
            # 计算F1
            if acc + recall > 0:
                f1 = 2 * acc * recall / (acc + recall)
                correctness_score = f1
            else:
                correctness_score = 0.0
            
            total_score = param_score + 2 * correctness_score
            
            if total_score > best_match_score:
                best_match_score = total_score
                best_match_index = i
        
        if best_match_index != -1:
            used_pd_indices.add(best_match_index)
            score += best_match_score
    
    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


def acc_reward(data_source, solution_str, ground_truth, extra_info=None):
    """Tool call正确性奖励函数"""
    # 固定奖励范围（不随step变化）

    max_possible_reward = 2.0
    min_possible_reward = 0
    
    # 处理输入：统一转换为列表
    if isinstance(solution_str, str):
        responses = [solution_str]
    else:
        responses = solution_str
    
    if isinstance(ground_truth, str):
        answers = [ground_truth]
    else:
        answers = ground_truth
    
    assert len(responses) == len(answers), f"responses和answers长度不匹配: {len(responses)} vs {len(answers)}"
    
    rewards = []
    
    for idx, (response, ans) in enumerate(zip(responses, answers)):
        print(f"\n{'='*50}")
        print(f"Processing sample {idx}")
        print(f"Response:\n{response[:200]}...")  # 只打印前200字符
        print(f"Ground truth: {ans}")

        try:
            # 解析ground truth
            gt_tools = json.loads(ans.strip())
            if not isinstance(gt_tools, list):
                raise ValueError("ground_truth解析后不是list")
            print(f"Parsed GT tools: {gt_tools}")

            # 检查格式
            if "<tool_calls>" not in response or "</tool_calls>" not in response:
                raise ValueError("未找到<tool_calls>或</tool_calls>标签")

            # 提取和解析predicted tools
            pd_tools_str = response.split("<tool_calls>")[1].split("</tool_calls>")[0].strip()
            print(f"Extracted tool_calls string: {pd_tools_str}")

            pd_tools = json.loads(pd_tools_str)
            if not isinstance(pd_tools, list):
                raise ValueError("预测tool_calls解析后不是list")
            print(f"Parsed PD tools: {pd_tools}")

            # 计算奖励
            reward = compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward)
            print(f"Computed reward: {reward}")
        except Exception as e:
            # 兜底：解析失败赋最小分，避免训练中断
            print(f"acc_reward warning: sample {idx} 解析失败，将赋最小分 {min_possible_reward}. error={e}")
            reward = min_possible_reward

        rewards.append(reward)
    
    print(f"\n{'='*50}")
    print("Final accuracy rewards:", rewards)
    return rewards


# ============= Length Reward =============
def extract_think(text):
    """提取<think>标签之间的内容"""
    if text is None:
        raise ValueError("extract_think收到了None值")
    
    match = re.search(r'<think>\n(.*?)\n</think>', text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        # 如果没有匹配到，返回整个文本
        print(f"Warning: 没有找到<think>标签，返回全部文本")
        return text.strip()


def len_reward(data_source, solution_str, ground_truth, extra_info=None):
    """长度奖励函数 - 简化版本，不考虑step"""
    # 固定参数
    center_l = 1024  # 理想最小长度
    center_r = 2048  # 最大允许长度
    sigma = 600      # 衰减速度
    min_val = 0.2    # 最小奖励值
    
    # 获取tokenizer
    tokenizer = get_tokenizer()
    if tokenizer is None:
        raise ValueError("Tokenizer未初始化，请设置BASE_MODEL环境变量")
    
    # 处理输入：统一转换为列表
    if isinstance(solution_str, str):
        completions = [solution_str]
    else:
        completions = solution_str
    
    rewards = []
    for idx, content in enumerate(completions):
        # 提取think内容
        think = extract_think(content)
        
        # 计算长度
        gen_len = len(tokenizer.encode(think))
        print(f"Sample {idx}: think长度 = {gen_len} tokens")
        
        # 计算奖励
        if gen_len <= center_l:
            reward = 1.0
        elif center_l < gen_len < center_r:
            # 使用高斯衰减
            reward = min_val + (1 - min_val) * math.exp(-((gen_len - center_l) ** 2) / (2 * sigma ** 2))
        else:  # gen_len >= center_r
            reward = 0.0
        
        print(f"Sample {idx}: length reward = {reward}")
        rewards.append(reward)
    
    print("Final length rewards:", rewards)
    return rewards


# ============= Soft Overlong Reward (可选) =============
def soft_overlong_reward(solution_str, soft_max_length=2048, soft_cache_length=512):
    """软性过长惩罚"""
    tokenizer = get_tokenizer()
    if tokenizer is None:
        raise ValueError("Tokenizer未初始化，请设置BASE_MODEL环境变量")
    
    assert soft_cache_length < soft_max_length, f"soft_cache_length ({soft_cache_length}) 必须小于 soft_max_length ({soft_max_length})"
    
    # 处理输入：统一转换为列表
    if isinstance(solution_str, str):
        completions = [solution_str]
    else:
        completions = solution_str
    
    rewards = []
    
    for completion in completions:
        completion_length = len(tokenizer.encode(completion))
        expected_len = soft_max_length - soft_cache_length
        exceed_len = completion_length - expected_len
        reward = min(-exceed_len / soft_cache_length, 0)
        rewards.append(reward)
    
    return rewards


# ============= 主计算函数 =============
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """综合评分函数 - 返回平均分"""
    print("\n" + "="*80)
    print("开始计算综合分数")
    print("="*80)
    
    res_format = format_reward(data_source, solution_str, ground_truth, extra_info)
    res_acc = acc_reward(data_source, solution_str, ground_truth, extra_info)
    res_len = len_reward(data_source, solution_str, ground_truth, extra_info)
    
    # 返回平均分
    total_score = (res_format[0] + res_acc[0] + res_len[0]) / 3.0
    
    print("\n" + "="*80)
    print(f"综合分数: {total_score}")
    print(f"  - Format: {res_format[0]}")
    print(f"  - Accuracy: {res_acc[0]}")
    print(f"  - Length: {res_len[0]}")
    print("="*80 + "\n")
    
    return total_score