import json
import os
import time
from typing import Any, Optional

import requests
from requests.exceptions import ConnectionError, ReadTimeout, SSLError
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    convert_to_function_call,
    convert_to_tool,
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    extract_system_prompt,
    format_execution_results_prompting,
    retry_with_backoff,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import contain_multi_turn_interaction

URL = os.getenv("CLAUDE_BEDROCK_PROXY_URL", "")
API_KEY = os.getenv("CLAUDE_BEDROCK_PROXY_API_KEY", "")


class ClaudeBedrockProxyHandler(BaseHandler):
    """
    Claude handler that calls a Bedrock/Anthropic-compatible proxy endpoint via `requests`.

    The proxy endpoint is expected to accept an Anthropic Messages-style payload, e.g.:
      {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 12800,
        "system": "...",
        "messages": [{"role": "user", "content": "..."}]
      }

    Required env vars:
      - CLAUDE_BEDROCK_PROXY_URL
      - CLAUDE_BEDROCK_PROXY_API_KEY
    Optional:
      - CLAUDE_BEDROCK_PROXY_ANTHROPIC_VERSION (default: bedrock-2023-05-31)
      - CLAUDE_BEDROCK_PROXY_MAX_TOKENS (default: 12800)
      - CLAUDE_BEDROCK_PROXY_TIMEOUT_SECS (default: 180)
    """

    def __init__(self, model_name: str, temperature: float) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.ANTHROPIC

        self._proxy_url = URL
        self._proxy_api_key = API_KEY
        self._anthropic_version = os.getenv(
            "CLAUDE_BEDROCK_PROXY_ANTHROPIC_VERSION", "bedrock-2023-05-31"
        )
        self._timeout_secs = int(os.getenv("CLAUDE_BEDROCK_PROXY_TIMEOUT_SECS", "180"))

        if not self._proxy_url:
            raise ValueError(
                "Missing env var `CLAUDE_BEDROCK_PROXY_URL` for ClaudeBedrockProxyHandler."
            )
        if not self._proxy_api_key:
            raise ValueError(
                "Missing env var `CLAUDE_BEDROCK_PROXY_API_KEY` for ClaudeBedrockProxyHandler."
            )

    def _get_max_tokens(self) -> int:
        return int(os.getenv("CLAUDE_BEDROCK_PROXY_MAX_TOKENS", "12800"))

    def _is_fc_mode(self) -> bool:
        return "FC" in self.model_name or self.is_fc_model

    def decode_ast(self, result, language, has_tool_call_tag):
        if self._is_fc_mode():
            decoded_output = []
            for invoked_function in result:
                name = list(invoked_function.keys())[0]
                params = json.loads(invoked_function[name])
                decoded_output.append({name: params})
            return decoded_output
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        if self._is_fc_mode():
            # In FC mode, Bedrock may still return plain text blocks on some steps.
            # Treat non-dict payloads as "no function call" instead of raising decode errors.
            if not isinstance(result, list):
                return []
            if any(not isinstance(item, dict) for item in result):
                return []
            return convert_to_function_call(result)
        return default_decode_execute_prompting(result, has_tool_call_tag)

    @retry_with_backoff(
        error_type=[SSLError, ConnectionError, ReadTimeout],
        error_message_pattern=r".*(429|Too Many Requests|rate limit).*",
    )
    def _post_with_backoff(self, payload: dict) -> dict:
        response = requests.post(
            url=self._proxy_url,
            headers={
                "Content-Type": "application/json",
                "api-key": self._proxy_api_key,
                "Connection": "close",
            },
            json=payload,
            timeout=self._timeout_secs,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Claude bedrock proxy error {response.status_code}: {response.text[:1000]}"
            )
        return response.json()

    #### FC methods ####

    def _query_FC(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "message": inference_data["message"],
            "tools": inference_data["tools"],
            "system_prompt": inference_data.get("system_prompt", []),
        }

        payload = {
            "anthropic_version": self._anthropic_version,
            "max_tokens": self._get_max_tokens(),
            "temperature": self.temperature,
            "messages": inference_data["message"],
            "tools": inference_data["tools"],
        }

        if "system_prompt" in inference_data:
            payload["system"] = inference_data["system_prompt"]

        start_time = time.time()
        api_response = self._post_with_backoff(payload)
        end_time = time.time()
        return api_response, end_time - start_time

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["message"] = []
        # Claude-style APIs pass system prompt in a dedicated field.
        system_prompt = extract_system_prompt(test_entry["question"][0])
        if system_prompt is not None:
            inference_data["system_prompt"] = system_prompt

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        test_category: str = test_entry["id"].rsplit("_", 1)[0]
        inference_data["caching_enabled"] = contain_multi_turn_interaction(test_category)
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        inference_data["tools"] = convert_to_tool(
            functions, GORILLA_TO_OPENAPI, self.model_style
        )
        return inference_data

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        content = api_response.get("content", []) if isinstance(api_response, dict) else []
        text_outputs = []
        tool_call_outputs = []
        tool_call_ids = []

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_outputs.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                tool_call_outputs.append({tool_name: json.dumps(tool_input)})
                tool_call_ids.append(block.get("id", ""))

        model_responses = tool_call_outputs if tool_call_outputs else text_outputs
        usage = api_response.get("usage", {}) if isinstance(api_response, dict) else {}
        input_tokens: Optional[int] = usage.get("input_tokens") or usage.get("inputTokens")
        output_tokens: Optional[int] = usage.get("output_tokens") or usage.get(
            "outputTokens"
        )

        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": content,
            "tool_call_ids": tool_call_ids,
            "input_token": input_tokens if input_tokens is not None else "N/A",
            "output_token": output_tokens if output_tokens is not None else "N/A",
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        for message in first_turn_message:
            if isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        for message in user_message:
            if isinstance(message.get("content"), str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses_message_for_chat_history"],
            }
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        tool_message = {
            "role": "user",
            "content": [],
        }
        for execution_result, tool_call_id in zip(
            execution_results, model_response_data["tool_call_ids"]
        ):
            tool_message["content"].append(
                {
                    "type": "tool_result",
                    "content": execution_result,
                    "tool_use_id": tool_call_id,
                }
            )
        inference_data["message"].append(tool_message)
        return inference_data

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict):
        inference_data["inference_input_log"] = {
            "system": inference_data.get("system_prompt", ""),
            "messages": inference_data["message"],
        }

        payload = {
            "anthropic_version": self._anthropic_version,
            "max_tokens": self._get_max_tokens(),
            "temperature": self.temperature,
            "system": inference_data.get("system_prompt", ""),
            "messages": inference_data["message"],
        }

        start_time = time.time()
        api_response = self._post_with_backoff(payload)
        end_time = time.time()

        return api_response, end_time - start_time

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]
        test_category: str = test_entry_id.rsplit("_", 1)[0]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        system_prompt = extract_system_prompt(test_entry["question"][0]) or ""

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        caching_enabled: bool = contain_multi_turn_interaction(test_category)

        return {
            "message": [],
            "system_prompt": system_prompt,
            "caching_enabled": caching_enabled,
        }

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        content = api_response.get("content")
        if isinstance(content, list) and content:
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            model_text = "".join(text_parts) if text_parts else str(content)
        elif isinstance(content, str):
            model_text = content
        else:
            model_text = api_response.get("completion") or api_response.get("output") or str(
                api_response
            )

        usage = api_response.get("usage", {}) if isinstance(api_response, dict) else {}
        input_tokens: Optional[int] = usage.get("input_tokens") or usage.get("inputTokens")
        output_tokens: Optional[int] = usage.get("output_tokens") or usage.get(
            "outputTokens"
        )

        return {
            "model_responses": model_text,
            "input_token": input_tokens if input_tokens is not None else "N/A",
            "output_token": output_tokens if output_tokens is not None else "N/A",
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            {
                "role": "assistant",
                "content": model_response_data["model_responses"],
            }
        )
        return inference_data

    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: list[str], model_response_data: dict
    ) -> dict:
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {
                "role": "user",
                "content": formatted_results_message,
            }
        )
        return inference_data
