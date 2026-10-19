import json
import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ==============================================================================
# 1) 错误示例库（沿用你提供的）
# ==============================================================================
ERROR_EXAMPLES = {
    "RETAIL / E-COMMERCE (tau-bench style)": [
        "ERROR: user not found",
        "ERROR: order not found",
        "ERROR: product not found",
        "ERROR: non-pending order cannot be cancelled",
        "ERROR: non-pending order cannot be modified",
        "ERROR: non-delivered order cannot be returned",
        "ERROR: payment method not found",
        "ERROR: insufficient gift card balance to pay for the order",
        "ERROR: the new payment method should be different from the current one",
        "ERROR: there should be exactly one payment for a pending order",
        "ERROR: some item not found",
        "ERROR: the number of items to be exchanged should match",
    ],
    "AIRLINE / TRAVEL (tau-bench + bfcl style)": [
        "ERROR: user not found",
        "ERROR: reservation not found",
        "ERROR: flight {flight_no} not found",
        "ERROR: flight {flight_no} not available on date {date}",
        "ERROR: not enough seats on flight {flight_no}",
        "ERROR: payment method {payment_id} not found",
        "ERROR: gift card balance is not enough",
        "ERROR: payment amount does not add up, total price is {total_price}, but paid {paid_total}",
        "ERROR: number of passengers does not match",
        "ERROR: Token not initialized",
        "ERROR: Invalid date format. Use YYYY-MM-DD.",
        "ERROR: Invalid departure airport code: {travel_from}",
    ],
    "FILE SYSTEM / OS (bfcl Gorilla style)": [
        "ERROR: cd: '{path}': No such file or directory",
        "ERROR: mkdir: cannot create directory '{dir_name}': File exists",
        "ERROR: touch: cannot touch '{file_name}': File exists",
        "ERROR: cat: '{file_name}': No such file or directory",
        "ERROR: cat: '{file_name}': Is a directory",
        "ERROR: mv: cannot move '{source}': No such file or directory",
        "ERROR: rm: cannot remove '{file_name}': No such file or directory",
        "ERROR: rmdir: cannot remove '{dir_name}': Directory not empty",
        "ERROR: grep: {file_name}: No such file or directory",
    ],
    "TRADING / SOCIAL / MESSAGING / TICKETS (bfcl style)": [
        "ERROR: User not authenticated. Please log in to place an order.",
        "ERROR: User not authenticated. Please log in to make a transaction.",
        "ERROR: No user is currently logged in.",
        "ERROR: Market is closed. Transactions are not allowed.",
        "ERROR: Insufficient funds for withdrawal.",
        "ERROR: Stock with symbol '{symbol}' not found.",
        "ERROR: Order with ID {order_id} not found.",
        "ERROR: Can't cancel order {order_id}. Order is already completed.",
        "ERROR: User '{user}' not found in the workspace.",
        "ERROR: Ticket with ID {ticket_id} not found.",
        "ERROR: Ticket with ID {ticket_id} is already closed.",
        "ERROR: Invalid priority. Priority must be between 1 and 5.",
    ],
    "VEHICLE / IOT (bfcl style)": [
        "ERROR: All doors must be locked before starting the engine.",
        "ERROR: Brake pedal needs to be pressed when starting the engine.",
        "ERROR: Start the engine before activating the cruise control.",
        "ERROR: Fuel tank is empty.",
        "ERROR: Cannot fill gas above the tank capacity.",
        "ERROR: Fan speed must be between 0 and 100.",
        "ERROR: Invalid headlight mode.",
        "ERROR: distance not found in database.",
    ],
    "MATH / GENERAL LOGIC (bfcl style)": [
        "ERROR: Cannot divide by zero",
        "ERROR: Both inputs must be numbers",
        "ERROR: All elements in the list must be numbers",
        "ERROR: Cannot calculate mean of an empty list",
        "ERROR: Cannot calculate square root of a negative number",
        "ERROR: Invalid date format. Use YYYY-MM-DD.",
        "ERROR: invalid characters in expression",
    ],
}


def format_error_examples_for_prompt(max_per_domain: int = 8) -> str:
    lines = []
    for domain, samples in ERROR_EXAMPLES.items():
        pick = samples[:max_per_domain]
        lines.append(f"\n**{domain}:**")
        for s in pick:
            lines.append(f"  • {s}")
    return "\n".join(lines)


def build_simulator_system_prompt() -> str:
    examples = format_error_examples_for_prompt(max_per_domain=8)
    return f"""You are a Runtime Environment Simulator for an AI Agent.
Your role is to act as the API Server or Operating System that executes tool calls.

**IMPORTANT CONTEXT:**
You are receiving a tool call from an Agent that has **ALREADY FAILED** validation or logic checks against the Ground Truth.
Your task is NOT to judge correctness. Your task is to **generate the specific ERROR MESSAGE** that the system would return to the Agent.

**GOAL:**
Generate a short, realistic, and actionable error message (starting with "ERROR: ") that will help the Agent understand **why** its call failed compared to the expected Ground Truth.

**--- PRIORITY ERROR CATEGORIES ---**
1. **Dependency & Sequence Violations**
2. **Parameter Hallucination**
3. **Schema & Parameter Errors**
4. **Business Logic Errors**

**--- EVALUATION LOGIC ---**
• **Reference the Ground Truth**: The ground_truth is usually correct and serves as the primary standard.
• **Verify Context**: Cross-check the Agent's output against the actual User Request in the Original Context.
• **Ambiguity Rule**: If the Agent's output differs from ground_truth but is still plausible, note it as missing validation.

**--- OUTPUT RULES ---**
1. **Start with "ERROR: "** (exactly this format, case-sensitive)
2. **Be Specific**: Reference actual parameter names and values from the failed attempt
3. **Sound like a system/API response**
4. **Keep it concise**: 1-2 sentences maximum
5. **Return ONLY the error string**: No JSON wrapping, no markdown, no explanation

**--- ERROR MESSAGE EXAMPLES (Real Benchmark Style) ---**
{examples}

**--- CRITICAL REMINDERS ---**
• Do NOT output JSON like {{"error": "..."}}, just output the plain text error message
• Do NOT add explanations or preambles, just the ERROR: line
• Do NOT use placeholder values like {{ticket_id}} unless they appear in the actual failed attempt
• The error should be what the RUNTIME SYSTEM returns, not your analysis of what went wrong"""


SIMULATOR_SYSTEM_PROMPT = build_simulator_system_prompt()


def build_simulation_input(original_system_content: str,
                           original_user_content: str,
                           ground_truth: str,
                           bad_attempt: str) -> str:
    return f"""## Simulation Task
The Agent attempted to execute a tool call, but it was **INCORRECT** compared to the Ground Truth.
Please generate the system error message triggered by the Agent's specific mistake.

### 1. Original Context (What the Agent saw)
**System Instructions & Tools:**
```text
{original_system_content}
```

**Conversation State Before This Attempt (Non-system history):**
```text
{original_user_content}
```

### 2. Execution Comparison
**The Correct Plan (Ground Truth):**
```json
{ground_truth}
```

**The Agent's Failed Attempt (Extracted Tool Call):**
```json
{bad_attempt}
```

### 3. Instructions for Simulator
Compare the **Agent's Attempt** against the **Correct Plan** and the **Original Context**.
Identify the **First Critical Failure** and generate the runtime error.

**Output:**
Return ONLY the error string starting with `ERROR:`."""


# ==============================================================================
# 2) raw_messages 解析：system + 全量非 system 历史
# ==============================================================================
def _to_list(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except Exception:
        try:
            return list(x.tolist())
        except Exception:
            return []


def extract_original_system_and_history(raw_messages: Any) -> Tuple[str, str]:
    msgs = _to_list(raw_messages)

    original_system = ""
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "system":
            original_system = m.get("content", "") or ""
            break

    # system 之外的全部消息，保留顺序，并带 role 标注（更稳）
    parts: List[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "")
        if role == "system":
            continue
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        parts.append(f"[{role}] {content}")

    history = "\n\n".join(parts).strip()
    return original_system, history


# ==============================================================================
# 3) bad_attempt 提取：沿用原反馈逻辑（<tool_calls>），并增强裸 JSON 数组
# ==============================================================================
def extract_tool_calls_from_completion(completion: str) -> Optional[str]:
    """
    兼容两种格式：
    1) 标准：<tool_calls> … </tool_calls>
    2) 你日志里常见：… 后面直接跟一个 JSON 数组（无标签）
    返回：JSON 数组字符串（可被 json.loads）
    """
    if not isinstance(completion, str):
        return None
    text = completion.strip()
    if not text:
        return None

    # 1) 原版逻辑：tool_calls 标签
    if "<tool_calls>" in text and "</tool_calls>" in text:
        try:
            inner = text.split("<tool_calls>", 1)[1].split("</tool_calls>", 1)[0].strip()
            obj = json.loads(inner)
            if isinstance(obj, list):
                return inner
        except Exception:
            pass

    # 2) 增强：从文本里找"最后一个可解析的 JSON 数组"
    # 先用正则收集候选，再倒序挑能 json.loads 的最后一个
    candidates = re.findall(r"(\[\s*\{[\s\S]*?\}\s*\])", text)
    for cand in reversed(candidates):
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, list):
                return cand
        except Exception:
            continue

    # 3) 兜底：括号配对，从最后一个 '[' 开始提取
    last_lbracket = text.rfind("[")
    if last_lbracket != -1:
        tail = text[last_lbracket:]
        depth = 0
        end = None
        for i, ch in enumerate(tail):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            cand = tail[:end].strip()
            try:
                obj = json.loads(cand)
                if isinstance(obj, list):
                    return cand
            except Exception:
                pass

    return None


# ==============================================================================
# 4) FeedbackLLM：主入口只收 raw_messages + wrong_answer + gt_tools
# ==============================================================================
class FeedbackLLM:
    FALLBACK_ERROR_MESSAGE = "ERROR: Function call failed. Please verify your output format, function name, required parameters, and parameter values are correct."


    def __init__(
        self,
        api_url: str = "https://your-openai-compatible-endpoint.example.com/v1",
        api_key: Optional[str] = None,
        model: str = "qwen3-235b-a22b-instruct-2507",
        timeout: int = 15,
        max_retries: int = 3,
        max_tokens: int = 160,
        temperature: float = 0.1,
        max_history_chars: int = 12000,
    ):
        self.api_url = api_url or os.getenv("XHS_MAAS_BASE_URL", "https://your-openai-compatible-endpoint.example.com/v1")
        self.api_key = api_key or os.getenv("XHS_MAAS_API_KEY") or os.getenv("BFCL_SIMULATOR_API_KEY")
        self.model = model or os.getenv("XHS_MAAS_MODEL", "qwen3-235b-a22b-instruct-2507")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_history_chars = max_history_chars

        self.client: Optional[OpenAI] = None
        if not self.api_key:
            logger.warning("[FeedbackLLM] API key not found; will fallback.")
        else:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
                logger.info(f"[FeedbackLLM] Initialized client base_url={self.api_url}, model={self.model}")
            except Exception as e:
                logger.error(f"[FeedbackLLM] Failed to init OpenAI client: {e}")
                self.client = None

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if self.client is None:
            return self.FALLBACK_ERROR_MESSAGE

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                if resp.choices and resp.choices[0].message and resp.choices[0].message.content:
                    return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                logger.error(f"[FeedbackLLM] LLM request failed: {e} (attempt {attempt+1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(1)

        logger.error(f"[FeedbackLLM] All retries failed. Last error: {last_err}")
        return self.FALLBACK_ERROR_MESSAGE

    @staticmethod
    def _postprocess_error(text: str, max_len: int = 260) -> str:
        if not isinstance(text, str):
            return FeedbackLLM.FALLBACK_ERROR_MESSAGE
        s = text.strip().replace("```", "").replace("\r", " ").replace("\n", " ")
        s = " ".join(s.split())

        # 抽第一条 ERROR:...
        m = re.search(r"(ERROR:\s*.+)", s)
        if m:
            s = m.group(1).strip()

        if not s.startswith("ERROR:"):
            if s.lower().startswith("error:"):
                s = "ERROR:" + s[6:]
            else:
                s = "ERROR: " + s

        if len(s) > max_len:
            s = s[:max_len].rstrip() + "..."
        return s

    def generate_feedback_from_trace(
        self,
        raw_messages: Any,
        wrong_answer: str,
        ground_truth_tools: str,
        request_id: str = "unknown",
    ) -> str:
        """
        你要的接口：
        - 从 raw_messages 抽 system + 非 system 全量历史
        - 从 wrong_answer 抽 bad_attempt（工具数组 JSON）
        - ground_truth_tools 由主代码从 reward_model['ground_truth'] 取出传入
        """
        logger.info(f"[FeedbackLLM-{request_id}] Generating feedback")

        original_system, history = extract_original_system_and_history(raw_messages)
        if len(history) > self.max_history_chars:
            history = history[: self.max_history_chars].rstrip() + "..."

        gt = (ground_truth_tools or "").strip()
        if not gt:
            gt = "[]"

        bad_attempt = extract_tool_calls_from_completion(wrong_answer)
        if bad_attempt is None:
            # 抽不到就退化为一个最小信息（避免塞 think）
            bad_attempt = "[]"

        human_input = build_simulation_input(
            original_system_content=original_system,
            original_user_content=history,
            ground_truth=gt,
            bad_attempt=bad_attempt,
        )

        messages = [
            {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
            {"role": "user", "content": human_input},
        ]

        raw = self._call_llm(messages)
        out = self._postprocess_error(raw)
        logger.info(f"[FeedbackLLM-{request_id}] Final error: {out}")
        return out
