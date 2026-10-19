import json
import os
import re
from typing import Any

from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.constants.enums import ModelStyle
from openai import OpenAI


class RedBookHandler(OpenAICompletionsHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        self.client = OpenAI(
            base_url=os.getenv("REDBOOK_BASE_URL"),
            api_key=os.getenv("REDBOOK_API_KEY"),
        )
        self.extra_body_from_env = self._load_extra_body_from_env()

    def _load_extra_body_from_env(self) -> dict:
        raw = os.getenv("REDBOOK_EXTRA_BODY_JSON", "").strip()
        if not raw:
            return {}

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid REDBOOK_EXTRA_BODY_JSON: {e}") from e

        if not isinstance(parsed, dict):
            raise ValueError(
                "REDBOOK_EXTRA_BODY_JSON must decode to a JSON object."
            )

        return parsed

    def _merge_dicts(self, base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def generate_with_backoff(self, **kwargs):
        extra_body = kwargs.get("extra_body", {}) or {}
        if self.extra_body_from_env:
            kwargs["extra_body"] = self._merge_dicts(
                extra_body, self.extra_body_from_env
            )
        return super().generate_with_backoff(**kwargs)

    def decode_ast(self, result, language, has_tool_call_tag):
        if "FC" in self.model_name or self.is_fc_model:
            return super().decode_ast(result, language, has_tool_call_tag)
        decoded_output = []
        for invoked_function in result:
            name = invoked_function["name"]
            params = invoked_function["arguments"]
            decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result, has_tool_call_tag):
        if "FC" in self.model_name or self.is_fc_model:
            return super().decode_execute(result, has_tool_call_tag)
        too_call_format = []
        for tool_call in result:
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})
                args_str = ", ".join(
                    [f"{key}={repr(value)}" for key, value in arguments.items()]
                )
                too_call_format.append(f"{name}({args_str})")
        return too_call_format

    #### Prompting methods ####

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        test_entry["question"][0] = self.redbook_system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_category
        )
        return {"message": []}

    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        message = api_response.choices[0].message
        content = message.content if isinstance(message.content, str) else ""
        tool_calls = content

        # Some providers return structured tool_calls with content=None.
        if content == "" and getattr(message, "tool_calls", None):
            tool_calls = []
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                tool_calls.append(
                    {"name": tool_call.function.name, "arguments": arguments}
                )
        else:
            match = re.search(
                r"<tool_calls>\n(.*?)\n</tool_calls>", content, re.DOTALL
            )
            if match:
                tool_calls = match.group(1).strip()
            try:
                tool_calls = tool_calls.replace("'", '"')
                tool_calls = json.loads(tool_calls)
            except Exception:
                pass

        return {
            "model_responses": tool_calls,
            "model_responses_message_for_chat_history": message,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    def redbook_system_prompt_pre_processing_chat_model(self,prompts, function_docs, test_category):
        system_pre="""You are an expert in composing functions.
You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

Here is a list of functions in JSON format that you can invoke.
<tools>
{}
</tools>
"""
        system_suffix = """
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the following format:
<think>
{reasoning process here}
</think>
<tool_calls>
[{"name": "func_name1", "arguments": {"param1": "value1", "param2": "value2"}}, {"name": "func_name2", "arguments": {"param": "value"}}]
</tool_calls>

You SHOULD NOT include any other text in the response when returning function calls.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""
#         system_pre="""You are a function calling AI model.  
# You are provided with function signatures within <tools></tools> XML tags.  
# You may call one or more functions to assist with the user query.  
# Don't make assumptions about what values to plug into functions.

# Today is 20250902.

# Here are the available tools:  
# <tools>  
# {}  
# </tools>
# """
#         system_suffix = """

# Use the following pydantic model json schema for each tool call you will make:  
# {"title": "FunctionCalls", "type": "array", "properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"]}

# # Output Format & Constraints

# At each turn, you should try your best to complete the user's request in the current turn.

# **Reasoning:**  
# You FIRST think about the reasoning process as an internal monologue, and then provide the final response. The reasoning MUST be enclosed within <think></think> tags.

# **Function Calls:**  
# - If you need to call any functions, output function calls within <tool_calls></tool_calls> tags.  
# - The entire content inside <tool_calls></tool_calls> MUST be a valid JSON array, where each item is a JSON object with "name" and "arguments" as specified by the schema.  
# - NEVER output <tool_calls> and <answer> tags at the same time; only one should appear per turn.  
# - Do NOT call tools if the question can be answered directly.

# **Final Answer:**  
# - If the question can be fully answered based on current information (without tools), use <answer></answer> tags.  
# - Inside <answer></answer>, provide only a short, precise answer to the question (not lengthy explanations).  
# - Even if you do not know the answer, output your answer inside <answer></answer> as a JSON:  
#   - {"answer": "I do not know", "context": "I do not know"}  
#   - If you cannot answer the question at all, output: {"answer": "I cannot answer this question", "context": "A short reason explaining why this question cannot be answered"}  
  
# **General Constraints:**  
# - At each turn, output ONLY ONE of: <tool_calls></tool_calls> OR <answer></answer> (never both).  
# - If you selected <answer></answer>, you MUST NOT propose another tool call even if the question is not answerable.  
# - All outputs must strictly follow the above format.  
# - Do not insert any additional explanation or commentary outside the specified tags.  
# - When using <tool_calls></tool_calls>, the JSON array must not be empty and must strictly conform to the schema above.  
# - Be careful not to misuse double quotes in the output json format.  
# - Tool Invocation Priority: During intermediate steps, if the final answer cannot yet be derived, you must continue invoking tools until sufficient information is obtained.  
  
# **Final Step Rule:**  
# - For multi-step reasoning tasks (e.g., web-search), the FINAL step MUST always end with an <answer> block.  
# - Once you output <answer>, you must never output <tool_calls> again.  
# - Even if the answer is uncertain or incomplete, you must still provide <answer> in the required format.  
  
# **Double-check Requirement:**  
# - Before producing the final <answer>, the model must perform a Double-Check step: re-verify all calculations step-by-step, validate factual claims or flag uncertainty, ensure logical consistency and completeness, and confirm the output follows the required format, then provide the corrected and validated final answer.  
  
# **Dynamic Plan Update:**  
# - During double-check, if issues or inconsistencies are found, you must update the plan in <think> and continue invoking tools until the problem is resolved, only then output the final answer.  
  
# **Attention**  
# If no suitable function is found, just respond with XML tags as follows,output your answer inside <answer></answer> as a JSON, don't use <tool_calls></tool_calls>  
# At any time, make sure that the <think></think> tag contains enough thoughts.  
  
# **Memory Management Instructions**
# When memory tools are available, you MUST call memory tools to manage user information - do not generate a direct response.

# **Critical Rules:**
# 1. Always use memory tools when provided, based on the user's latest message
# 2. NEVER clear existing memory unless explicitly requested by the user
# 3. Use replace to update outdated information, or append to add new content
# 4. Summarize and extract key points - do not copy the user's words verbatim

# **Memory Compression (when needed):**
# If you encounter capacity warnings (e.g., exceeding 10000 characters):
# 1. First use memory_retrieve to check current stored information
# 2. Then use memory_update to compress and consolidate memories
# 3. Remove redundancy and duplicates while preserving all key information
# 4. WARNING: memory_update replaces ALL existing memory - use with extreme caution and ensure comprehensive consolidation

# **Storage Format:**
# Use clear topic headers with bullet points:

# [User Profile]
# - Name, location, profession, key context

# [Preferences & Requirements]
# - Key preferences and constraints by category

# [Interaction History]
# - Notable decisions, ongoing projects, and pain points

# Keep content concise, relevant, and avoid redundancy with existing memory.

# """
        assert type(prompts) == list

        system_prompt = system_pre.format(function_docs)+system_suffix

        # System prompt must be in the first position
        # If the question comes with a system prompt, append its content at the end of the chat template.
        if prompts[0]["role"] == "system":
            prompts[0]["content"] = system_prompt + "\n\n" + prompts[0]["content"]
        # Otherwise, use the system prompt template to create a new system prompt.
        else:
            prompts.insert(
                0,
                {"role": "system", "content": system_prompt},
            )

        return prompts
