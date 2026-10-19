# Copyright (c) Alibaba, Inc. and its affiliates. {#copyright-c-alibaba-inc-and-its-affiliates  data-source-line="1303"}
"""
Fission Feedback Module: LLM-based Error Simulator for GRPO Training

This module extracts error simulation logic from plugin_simulate_v3.py, providing
a clean interface for generating error feedback messages that guide model correction
in the fission training workflow.

The error simulator analyzes incorrect tool calls and generates specific feedback
following BFCL (Berkeley Function Calling Leaderboard) error patterns.
"""
import json
import os
import re
import time
import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

class FeedbackLLM:
    """
    LLM-based error feedback generator for tool calling errors.

    This class encapsulates the logic for:
    1. Calling an external LLM API to generate error feedback
    2. Parsing and validating BFCL-compatible error messages
    3. Extracting tool call information from completions

    Usage:
        feedback_llm = FeedbackLLM(api_url="http://localhost:8399/v1/chat/completions",
                                    api_key="sk-xxx", model="qwen3_30b")
        error_msg = feedback_llm.generate_feedback(
            problem_description="Calculate the area of a circle",
            api_docs="...",
            ground_truth_tools=[{"name": "calculate_area", "arguments": {"radius": 5}}],
            predicted_tools=[{"name": "calculate_volume", "arguments": {"radius": 5}}]
        )
    """

    # System prompt for the error simulator (from plugin_simulate_v3.py)
    SIMULATOR_SYSTEM_PROMPT = """
You are a BFCL (Berkeley Function Calling Leaderboard) System Environment Simulator operating in "Prompting Mode".

Your task is to simulate realistic system error responses that would be returned when an agent makes an incorrect function call. These errors will be injected as user messages in the format used during actual BFCL testing.

**--- CRITICAL: Understanding Prompting Mode ---**

In Prompting Mode, execution results are formatted as Python-style list strings, for example:
``` {data-source-line="1355"}
[
  {"role": "tool", "name": "function_name(param='value')", "content": "{\\"error\\": \\"Error message here\\"}"}
]
```

Your job is to generate the ERROR MESSAGE that would appear in the "content" field, matching the exact patterns used by real BFCL testing systems.

**--- Real BFCL Error Message Patterns ---**

**1. GorillaFileSystem (文件系统):**
- Directory: "cd: {path}: No such directory", "Current directory is already the root. Cannot go back."
- Create: "mkdir: cannot create directory '{name}': File exists", "touch: cannot touch '{name}': File exists"
- Read: "cat: {file}: No such file or directory", "cat: {file}: Is a directory"
- Move/Copy: "mv: cannot move '{source}': No such file or directory", "cp: cannot copy '{source}': File exists"
- Delete: "rm: cannot remove '{file}': No such file or directory", "rmdir: {dir}: Directory not empty"
- Invalid chars: "mkdir: cannot create directory '{name}': Invalid character"

**2. TradingBot (交易系统):**
- Auth: "User not authenticated. Please log in to place an order/view account information/fund the account."
- Market: "Market is closed. Transactions are not allowed.", "Invalid stock symbol: {symbol}"
- Stock: "Stock with symbol '{symbol}' not found.", "Stock {symbol} not found in watchlist."
- Order: "Order with ID {order_id} not found.", "Can't cancel order {order_id}. Order is already completed."
- Funds: "Funding amount must be positive.", "Transaction amount must be positive."

**3. VehicleControlAPI (车载控制):**
- Engine: "All doors must be locked before starting the engine. Here are the unlocked doors: {doors}"
- Brake: "Brake pedal needs to be pressed when starting the engine.", "Must press the brake fully before starting the engine."
- Fuel: "Fuel tank is empty.", "Cannot fill gas above the tank capacity.", "Fuel amount cannot be negative."
- Systems: "Fan speed must be between 0 and 100.", "Invalid headlight mode.", "Pedal position must be between 0 and 1."
- Navigation: "distance not found in database."

**4. TravelAPI (旅行预订):**
- Token: "Token not initialized", "Token expired", "Invalid access token"
- Card: "Card already registered", "Card not registered. Here are a list of card_id's: {ids}", "Credit card not registered"
- Balance: "Balance not found", "Insufficient funds", "Balance is less than budget limit"
- Booking: "Booking not found", "Exceeded budget limit"

**5. TwitterAPI (社交媒体):**
- Auth: "User not authenticated. Please authenticate before posting/retweeting/following/unfollowing."
- Content: "Tweet with ID {tweet_id} not found."
- Special format: "Error during execution: {'error': 'User not authenticated. Please authenticate before commenting.'}"

**6. MessageAPI (消息系统):**
- User: "User '{user}' not found in the workspace.", "No user is currently logged in."
- Receiver: "Receiver ID '{receiver_id}' not found.", "Receiver ID {receiver_id} not found."
- Duplicate: "User name '{user_name}' already exists.", "User ID '{user_id}' already exists."

**7. TicketAPI (工单系统):**
- Auth: "User not authenticated. Please log in to create a ticket/view tickets."
- Priority: "Invalid priority. Priority must be between 1 and 5."
- Status: "Ticket with ID {ticket_id} not found.", "Ticket with ID {ticket_id} is already closed/resolved."
- Update: "Invalid fields for update: {fields}"

**8. MathAPI (数学计算):**
- Type: "Value must be a number", "Both inputs must be numbers", "All elements in the list must be numbers"
- Operation: "Cannot divide by zero", "Cannot calculate square root of a negative number"
- Empty: "Cannot calculate mean/standard deviation/sum of an empty list", "Cannot find minimum/maximum of an empty list"
- Conversion: "Conversion from '{unit_in}' to '{unit_out}' is not supported"

**--- Analysis Process ---**

Step 1: Identify which API/system the ground truth is calling
Step 2: Compare agent's call with ground truth to find the exact mistake
Step 3: Select the appropriate error pattern from the list above
Step 4: Generate a specific error message using that pattern

**--- Response Format ---**

Your response MUST follow this exact structure:

``` {data-source-line="1426"}
<classification>
API_SYSTEM: [GorillaFileSystem | TradingBot | VehicleControlAPI | TravelAPI | TwitterAPI | MessageAPI | TicketAPI | MathAPI]
ERROR_TYPE: [Wrong function | Missing parameter | Invalid value | Type mismatch | Auth failure | Resource not found | State violation]
SPECIFIC_DETAILS: [What exactly went wrong]
</classification>

[Exact error message using the patterns above - NO "Error:" prefix, just the raw message]
```

**--- Critical Rules ---**

1. **EXACT PATTERN MATCHING**: Use the EXACT error message patterns from the list above, only replacing {placeholders} with actual values
2. **NO CREATIVE WRITING**: Do not invent new error messages - only use the documented patterns
3. **SYSTEM-LIKE**: Messages should sound like they come from a system, not a human assistant
4. **SPECIFIC VALUES**: Replace placeholders like {symbol}, {file}, {id} with the actual values from the agent's call
5. **NO PREFIX**: Do NOT add "Error:" prefix - the message itself is the error

**--- Examples ---**

Example 1 - Wrong Function (File System):
``` {data-source-line="1447"}
Agent tried: delete_file(filename="report.txt")
Ground truth: rm(file_name="report.txt")

<classification>
API_SYSTEM: GorillaFileSystem
ERROR_TYPE: Wrong function
SPECIFIC_DETAILS: Agent called 'delete_file' which doesn't exist; should use 'rm'
</classification>

rm: cannot remove 'report.txt': No such file or directory
```

Example 2 - Missing Authentication (Trading):
``` {data-source-line="1461"}
Agent tried: get_stock_price(symbol="AAPL")
Ground truth: authenticate(username="user", password="pass") then get_stock_price(symbol="AAPL")

<classification>
API_SYSTEM: TradingBot
ERROR_TYPE: Auth failure
SPECIFIC_DETAILS: Agent tried to query stock without authenticating first
</classification>

User not authenticated. Please log in to view account information.
```

Example 3 - Invalid Parameter Value (Vehicle):
``` {data-source-line="1475"}
Agent tried: set_fan_speed(speed=150)
Ground truth: set_fan_speed(speed=75)

<classification>
API_SYSTEM: VehicleControlAPI
ERROR_TYPE: Invalid value
SPECIFIC_DETAILS: Fan speed 150 exceeds maximum allowed value of 100
</classification>

Fan speed must be between 0 and 100.
```

Example 4 - Missing Parameter (Travel):
``` {data-source-line="1489"}
Agent tried: register_card(card_number="1234-5678-9012-3456")
Ground truth: register_card(access_token="abc123", card_number="1234-5678-9012-3456")

<classification>
API_SYSTEM: TravelAPI
ERROR_TYPE: Missing parameter
SPECIFIC_DETAILS: Required parameter 'access_token' is missing
</classification>

Invalid access token
```
"""

    # BFCL error validation patterns (from plugin_simulate_v3.py)
    BFCL_PATTERNS = [
        # File system errors
        r"cd:.*No such (file or )?directory",
        r"mkdir:.*cannot create directory",
        r"cat:.*No such file or directory",
        r"cat:.*Is a directory",
        r"rm:.*cannot remove",
        r"mv:.*cannot move",
        r"cp:.*cannot copy",
        r".*: File exists",
        r".*: Invalid character",
        r".*Directory not empty",
        # Trading system errors
        r"Stock with symbol.*not found",
        r"Invalid stock symbol",
        r"Order with ID.*not found",
        r"Can't cancel order",
        r"User not authenticated.*Please log in",
        r"Market is closed",
        r"Insufficient funds",
        r".*amount must be positive",
        # Vehicle control errors
        r"All doors must be locked",
        r"Brake pedal needs to be pressed",
        r"Fuel tank is empty",
        r"Cannot fill gas above",
        r"Fan speed must be between",
        r"Invalid.*mode",
        r".*must be between \d+ and \d+",
        r"distance not found in database",
        # Travel API errors
        r"Token (not initialized|expired)",
        r"Invalid access token",
        r"Card (not registered|already registered)",
        r"Balance (not found|is less than)",
        r"Insufficient funds",
        r"Booking not found",
        r"Exceeded budget limit",
        # Social media errors
        r"User not authenticated.*authenticate before",
        r"Tweet with ID.*not found",
        # Message API errors
        r"User.*not found in the workspace",
        r"No user is currently logged in",
        r"Receiver ID.*not found",
        r".*already exists",
        # Ticket API errors
        r"Invalid priority.*must be between",
        r"Ticket with ID.*not found",
        r"Ticket with ID.*is already (closed|resolved)",
        r"Invalid fields for update",
        # Math API errors
        r"Value must be a number",
        r"Cannot divide by zero",
        r"Cannot calculate.*of (a negative number|an empty list)",
        r"All elements.*must be numbers",
        r"Conversion from.*to.*not supported",
    ]

    FALLBACK_ERROR_MESSAGE = (
        "Function call failed. Please verify your function name, "
        "required parameters, and parameter values are correct."
    )

    def __init__(
        self,
        api_url: str = "https://your-openai-compatible-endpoint.example.com/v1",
        api_key: Optional[str] = None,
        model: str = "qwen3-235b-a22b-instruct-2507",
        timeout: int = 15,
        max_retries: int = 3,
        enable_data_collection: bool = False,
        data_collection_file: str = "bfcl_simulator_training_data.jsonl"
    ):
        """
        Initialize the feedback LLM.

        Args:
            api_url: Base URL of the LLM API endpoint (OpenAI-compatible)
            api_key: API key for authentication (can be set via environment variable)
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            enable_data_collection: Whether to save simulator I/O for training
            data_collection_file: Path to save collected data
        """
        # XHS DirectLLM / OpenAI-compatible client configuration
        # 优先使用显式传入的参数，其次读取环境变量，最后回退到默认值。
        self.api_url = api_url or os.getenv("XHS_MAAS_BASE_URL", "https://your-openai-compatible-endpoint.example.com/v1")
        self.api_key = api_key or os.getenv("XHS_MAAS_API_KEY") or os.getenv("BFCL_SIMULATOR_API_KEY")
        self.model = model or os.getenv("XHS_MAAS_MODEL", "qwen3-235b-a22b-instruct-2507")
        self.timeout = timeout
        self.max_retries = max_retries

        self.enable_data_collection = enable_data_collection or os.getenv("BFCL_COLLECT_DATA", "0") == "1"
        self.data_collection_file = data_collection_file

        if self.enable_data_collection:
            logger.info(f"[FeedbackLLM] Data collection enabled, saving to: {self.data_collection_file}")

        # 初始化 OpenAI 客户端（用于访问小红书 DirectLLM）
        self.client: Optional[OpenAI] = None
        if not self.api_key:
            logger.warning(
                "[FeedbackLLM] OpenAI client or API key not available. "
                "Please ensure 'openai' is installed and XHS_MAAS_API_KEY (or BFCL_SIMULATOR_API_KEY) is set "
                "if you want to enable LLM-based feedback."
            )
        else:
            try:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
                logger.info(f"[FeedbackLLM] OpenAI client initialized with base_url={self.api_url}, model={self.model}")
            except Exception as e:
                logger.error(f"[FeedbackLLM] Failed to initialize OpenAI client: {e}")
                self.client = None

    def _call_llm_api(self, messages: list) -> str:
        """
        Call the LLM API with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            str: The LLM response content
        """
        if self.client is None:
            logger.error("[FeedbackLLM] OpenAI client is not initialized, using fallback message")
            return self.FALLBACK_ERROR_MESSAGE

        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                    # 错误信息不需要很长，这里限制一个较小的 max_tokens
                    max_tokens=256,
                    temperature=0.1,
                )
                # OpenAI ChatCompletion 格式：choices[0].message.content
                if not completion.choices:
                    logger.warning(
                        f"[FeedbackLLM] Empty choices in completion (attempt {attempt + 1}/{self.max_retries})"
                    )
                else:
                    content = completion.choices[0].message.content or ""
                    if content:
                        logger.debug(
                            f"[FeedbackLLM] LLM API call successful (attempt {attempt + 1}/{self.max_retries})"
                        )
                        return content.strip()
                    logger.warning(
                        f"[FeedbackLLM] Empty content in completion (attempt {attempt + 1}/{self.max_retries})"
                    )

            except Exception as e:
                logger.error(f"[FeedbackLLM] LLM request failed: {e} (attempt {attempt + 1}/{self.max_retries})")

            if attempt < self.max_retries - 1:
                time.sleep(1)

        logger.error(f"[FeedbackLLM] All {self.max_retries} attempts failed, using fallback message")
        return self.FALLBACK_ERROR_MESSAGE

    def _validate_bfcl_error(self, error_message: str) -> str:
        """
        Validate that the error message matches BFCL patterns.

        Args:
            error_message: The error message to validate

        Returns:
            str: The validated (or fallback) error message
        """
        # Check if matches any BFCL pattern
        for pattern in self.BFCL_PATTERNS:
            if re.search(pattern, error_message, re.IGNORECASE):
                logger.debug(f"[FeedbackLLM] Error message matches BFCL pattern: {pattern}")
                return error_message

        logger.warning(f"[FeedbackLLM] Error message does not match known BFCL patterns: {error_message[:100]}")

        # Check if message looks like an error (has error keywords)
        error_keywords = ['not', 'cannot', 'invalid', 'must', 'error', 'failed',
                          'missing', 'required', 'found', 'incorrect', 'wrong',
                          'unexpected', 'exists']

        if len(error_message) < 10 or not any(kw in error_message.lower() for kw in error_keywords):
            logger.warning("[FeedbackLLM] Message too short or lacks error keywords, using fallback")

            # Choose more specific fallback based on content
            lower_msg = error_message.lower()
            if 'parameter' in lower_msg or 'argument' in lower_msg:
                return ("Invalid parameter. Please check the function signature and provide "
                        "all required parameters with correct values.")
            elif 'function' in lower_msg or 'method' in lower_msg or 'call' in lower_msg:
                return ("Invalid function call. Please verify the function name and available "
                        "functions in the API documentation.")
            else:
                return self.FALLBACK_ERROR_MESSAGE

        return error_message

    def _save_training_data(self, data_entry: dict):
        """Save training data to JSONL file."""
        if not self.enable_data_collection:
            return

        try:
            with open(self.data_collection_file, 'a', encoding='utf-8') as f:
                json.dump(data_entry, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            logger.error(f"[FeedbackLLM] Failed to save training data: {e}")

    @staticmethod
    def extract_tool_calls_from_completion(completion: str) -> Optional[str]:
        """
        Extract tool calls JSON from completion text.

        Args:
            completion: The model's completion text

        Returns:
            str: Extracted JSON string, or None if not found
        """
        if "<tool_calls>" not in completion or "</tool_calls>" not in completion:
            return None

        try:
            tool_calls_content = completion.split("<tool_calls>")[1].split("</tool_calls>")[0].strip()
            return tool_calls_content
        except Exception as e:
            logger.warning(f"[FeedbackLLM] Failed to extract tool_calls: {e}")
            return None

    def generate_feedback(
        self,
        problem_description: str,
        api_docs: str,
        ground_truth_tools: str,  # JSON string
        predicted_completion: str,
        request_id: str = "unknown"
    ) -> str:
        """
        Generate error feedback for an incorrect tool call.

        Args:
            problem_description: The user's original problem/question
            api_docs: Available API/function documentation
            ground_truth_tools: JSON string of correct tool calls
            predicted_completion: The model's incorrect completion
            request_id: Optional request ID for logging

        Returns:
            str: The error feedback message to be shown to the model
        """
        logger.info(f"[FeedbackLLM-{request_id}] Generating error feedback")

        # Extract tool calls from the predicted completion
        parsed_predicted = self.extract_tool_calls_from_completion(predicted_completion)
        if parsed_predicted is None:
            parsed_predicted = predicted_completion
            logger.debug(f"[FeedbackLLM-{request_id}] Could not extract tool_calls, using full completion")

        # Construct the user prompt for the simulator
        user_prompt = f"""
## Task: Analyze Agent's Failed Function Call

### 1. What the Agent Was Trying to Accomplish:
{problem_description}

### 2. Available Functions/APIs:
{api_docs}

### 3. What the Agent Should Have Done (Ground Truth):
```json
{ground_truth_tools}
```

### 4. What the Agent Actually Did (Incorrect):
```json
{parsed_predicted}
```

### 5. Your Job:
Compare sections 3 and 4 carefully. Identify the EXACT mistake:
- Did the agent call the wrong function name?
- Did the agent miss required parameters?
- Did the agent use incorrect parameter values?
- Did the agent use wrong parameter types?
- Did the agent include unnecessary parameters?

Then generate a specific, actionable error message following the BFCL format defined in your system prompt.
"""

        # Call the LLM API
        raw_output = self._call_llm_api(
            messages=[
                {"role": "system", "content": self.SIMULATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract classification info (for logging)
        classification_match = re.search(r'<classification>(.*?)</classification>', raw_output, re.DOTALL)
        if classification_match:
            classification_info = classification_match.group(1).strip()
            logger.info(f"[FeedbackLLM-{request_id}] Error classification:\n{classification_info}")

        # Extract error message (after </classification> tag)
        if '</classification>' in raw_output:
            error_message = raw_output.split('</classification>')[-1].strip()
            error_message = error_message.replace('```', '').strip()
        else:
            error_message = raw_output.strip()

        # Validate against BFCL patterns
        final_feedback = self._validate_bfcl_error(error_message)

        logger.info(f"[FeedbackLLM-{request_id}] Final feedback: {repr(final_feedback)}")

        # Save training data if enabled
        if self.enable_data_collection:
            training_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": request_id,
                "simulator_input": {
                    "messages": [
                        {"role": "system", "content": self.SIMULATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ]
                },
                "simulator_output": raw_output,
                "final_feedback": final_feedback
            }
            self._save_training_data(training_data)

        return final_feedback
