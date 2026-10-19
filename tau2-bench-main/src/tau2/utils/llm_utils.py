import json
import random
import re
import threading
import time
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True
# Reduce repetitive LiteLLM helper spam; we log normalized errors ourselves.
litellm.suppress_debug_info = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


class _RequestKeyPool:
    """Round-robin pool with per-key in-flight limits."""

    def __init__(self, keys: list[str], per_key_max_inflight: int):
        self.keys = keys
        self._semaphores = [
            threading.Semaphore(max(1, per_key_max_inflight)) for _ in keys
        ]
        self._idx = 0
        self._idx_lock = threading.Lock()

    def next_start_index(self) -> int:
        with self._idx_lock:
            idx = self._idx
            self._idx = (self._idx + 1) % len(self.keys)
            return idx

    def acquire(self, key_index: int, timeout_seconds: float) -> bool:
        return self._semaphores[key_index].acquire(timeout=timeout_seconds)

    def release(self, key_index: int) -> None:
        self._semaphores[key_index].release()


_REQUEST_KEY_POOLS: dict[tuple[str, tuple[str, ...], int], _RequestKeyPool] = {}
_REQUEST_KEY_POOLS_LOCK = threading.Lock()


class _InvalidAssistantResponseError(RuntimeError):
    """Raised when model returns an assistant message without content/tool calls."""


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _parse_api_keys(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [
            v.strip()
            for v in re.split(r"[,\n]", raw_value)
            if isinstance(v, str) and v.strip()
        ]
    if isinstance(raw_value, (list, tuple)):
        parsed = []
        for item in raw_value:
            if item is None:
                continue
            item_str = str(item).strip()
            if item_str:
                parsed.append(item_str)
        return parsed
    return []


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"1", "true", "yes", "y", "on"}:
            return True
        if val in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, _InvalidAssistantResponseError):
        return True

    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
            return True
        if 400 <= status_code < 500:
            return False

    text = str(error).lower()
    if any(
        token in text
        for token in [
            "ratelimit",
            "rate limit",
            "quota",
            "429",
            "timeout",
            "timed out",
            "connection refused",
            "connection reset",
            "connection error",
            "service unavailable",
            "temporarily unavailable",
            "bad gateway",
            "gateway timeout",
            "internal server error",
        ]
    ):
        return True
    if any(
        token in text
        for token in [
            "badrequesterror",
            "invalid_request_error",
            "context_length_exceeded",
            '"code": 400',
        ]
    ):
        return False
    return False


def _assert_valid_assistant_response(response: ModelResponse) -> None:
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
        tool_calls = message.tool_calls
    except Exception as e:
        raise _InvalidAssistantResponseError(
            f"Malformed response payload: {e}"
        ) from e

    has_content = isinstance(content, str) and content.strip() != ""
    has_tool_calls = tool_calls is not None and len(tool_calls) > 0
    if not (has_content or has_tool_calls):
        raise _InvalidAssistantResponseError(
            "Model returned empty assistant message (no content and no tool_calls)."
        )


def _is_context_window_exceeded_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in [
            "contextwindowexceedederror",
            "maximum context length",
            "max context length",
            "max_tokens' or 'max_completion_tokens' is too large",
            "context length exceeded",
            "too many tokens",
            "input tokens",
        ]
    )


def _extract_context_budget(error: Exception) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Try to parse max context and input token counts from provider error text.
    Returns (max_context, input_tokens, available_output_tokens).
    """
    text = str(error)
    m = re.search(
        r"maximum context length is\s+(\d+)\s+tokens\s+and your request has\s+(\d+)\s+input tokens",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None, None
    max_context = int(m.group(1))
    input_tokens = int(m.group(2))
    return max_context, input_tokens, max_context - input_tokens


def _truncate_oldest_non_system_messages(
    litellm_messages: list[dict], drop_messages: int
) -> tuple[list[dict], int]:
    """
    Drop the oldest non-system messages while preserving all system prompts.
    """
    if drop_messages <= 0:
        drop_messages = 1
    new_messages = []
    removed = 0
    for msg in litellm_messages:
        role = msg.get("role")
        if role != "system" and removed < drop_messages:
            removed += 1
            continue
        new_messages.append(msg)
    return new_messages, removed


def _get_request_key_pool(
    api_base: str, keys: list[str], per_key_max_inflight: int
) -> _RequestKeyPool:
    pool_key = (api_base, tuple(keys), per_key_max_inflight)
    with _REQUEST_KEY_POOLS_LOCK:
        pool = _REQUEST_KEY_POOLS.get(pool_key)
        if pool is None:
            pool = _RequestKeyPool(keys=keys, per_key_max_inflight=per_key_max_inflight)
            _REQUEST_KEY_POOLS[pool_key] = pool
        return pool


def _completion_with_request_pool(
    model: str,
    litellm_messages: list[dict],
    tools: Optional[list[dict]],
    tool_choice: Optional[str],
    kwargs: dict[str, Any],
) -> ModelResponse:
    request_kwargs = dict(kwargs)

    api_keys = _parse_api_keys(request_kwargs.pop("api_keys", None))
    api_keys += _parse_api_keys(request_kwargs.pop("api_keys_csv", None))
    api_keys = _dedupe_keep_order(api_keys)

    pool_enabled = _to_bool(
        request_kwargs.pop("request_key_pool_enabled", None),
        default=len(api_keys) > 1,
    )
    if not pool_enabled or not api_keys:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **request_kwargs,
        )
        _assert_valid_assistant_response(response)
        return response

    per_key_max_inflight = max(
        1,
        _to_int(
            request_kwargs.pop(
                "request_per_key_max_inflight",
                request_kwargs.pop("per_key_max_inflight", 1),
            ),
            1,
        ),
    )
    request_max_retries = max(
        1,
        _to_int(
            request_kwargs.pop("request_max_retries", DEFAULT_MAX_RETRIES),
            DEFAULT_MAX_RETRIES,
        ),
    )
    request_base_delay = max(
        0.0,
        _to_float(request_kwargs.pop("request_base_delay", 1.0), 1.0),
    )
    request_max_delay = max(
        0.0,
        _to_float(request_kwargs.pop("request_max_delay", 30.0), 30.0),
    )
    request_jitter = max(
        0.0,
        _to_float(request_kwargs.pop("request_jitter", 0.5), 0.5),
    )
    key_acquire_timeout = max(
        1.0,
        _to_float(request_kwargs.pop("request_key_acquire_timeout", 120.0), 120.0),
    )

    api_base = str(request_kwargs.get("api_base", ""))
    pool = _get_request_key_pool(
        api_base=api_base,
        keys=api_keys,
        per_key_max_inflight=per_key_max_inflight,
    )
    start_index = pool.next_start_index()
    request_kwargs.pop("api_key", None)
    # We handle retries at request-level to avoid nested retry loops.
    request_kwargs["num_retries"] = 0

    last_error: Optional[Exception] = None
    key_count = len(api_keys)

    for hop in range(key_count):
        key_index = (start_index + hop) % key_count
        if not pool.acquire(key_index, timeout_seconds=key_acquire_timeout):
            continue
        try:
            for attempt in range(request_max_retries):
                call_kwargs = dict(request_kwargs)
                call_kwargs["api_key"] = api_keys[key_index]
                try:
                    response = completion(
                        model=model,
                        messages=litellm_messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        **call_kwargs,
                    )
                    _assert_valid_assistant_response(response)
                    return response
                except Exception as e:
                    last_error = e
                    if not _is_retryable_error(e):
                        raise
                    if attempt >= request_max_retries - 1:
                        break
                    sleep_seconds = request_base_delay * (2**attempt)
                    if request_max_delay > 0:
                        sleep_seconds = min(sleep_seconds, request_max_delay)
                    sleep_seconds += random.uniform(0.0, request_jitter)
                    logger.warning(
                        f"LLM request failed on key index {key_index} "
                        f"(attempt {attempt + 1}/{request_max_retries}), retrying in {sleep_seconds:.2f}s: "
                        f"{_format_exception(e, max_chars=300)}"
                    )
                    time.sleep(sleep_seconds)
        finally:
            pool.release(key_index)

    if last_error is not None:
        raise last_error
    raise RuntimeError("No available key in request key pool.")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def _format_exception(error: Exception, max_chars: int = 800) -> str:
    status_code = getattr(error, "status_code", None)
    body = str(error).replace("\n", " ").strip()
    if len(body) > max_chars:
        body = body[:max_chars] + "...(truncated)"
    if isinstance(status_code, int):
        return f"{type(error).__name__}(status_code={status_code}): {body}"
    return f"{type(error).__name__}: {body}"


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        # Some custom/openai-compatible models (e.g. glm-5) are not in LiteLLM's
        # pricing map. In that case we still run normally and just keep cost as 0.0.
        if "isn't mapped yet" in str(e):
            return 0.0
        logger.error(_format_exception(e))
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES
    auto_truncate_context = _to_bool(kwargs.pop("auto_truncate_context", True), True)
    context_truncate_max_retries = max(
        0, _to_int(kwargs.pop("context_truncate_max_retries", 8), 8)
    )
    context_truncate_drop_messages = max(
        1, _to_int(kwargs.pop("context_truncate_drop_messages", 2), 2)
    )
    context_safety_margin = max(
        0, _to_int(kwargs.pop("context_safety_margin", 128), 128)
    )
    context_min_output_tokens = max(
        1, _to_int(kwargs.pop("context_min_output_tokens", 64), 64)
    )

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"

    truncate_attempt = 0
    while True:
        try:
            response = _completion_with_request_pool(
                model=model,
                litellm_messages=litellm_messages,
                tools=tools,
                tool_choice=tool_choice,
                kwargs=kwargs,
            )
            break
        except Exception as e:
            if (
                auto_truncate_context
                and _is_context_window_exceeded_error(e)
                and truncate_attempt < context_truncate_max_retries
            ):
                # First try to shrink output budget before dropping conversation history.
                _, _, available_output = _extract_context_budget(e)
                cur_max_tokens = _to_int(kwargs.get("max_tokens"), 0)
                if available_output is not None and cur_max_tokens > 0:
                    target_max_tokens = min(
                        cur_max_tokens,
                        max(context_min_output_tokens, available_output - context_safety_margin),
                    )
                    if target_max_tokens < cur_max_tokens:
                        kwargs["max_tokens"] = target_max_tokens
                        truncate_attempt += 1
                        logger.warning(
                            f"Context window exceeded; lowered max_tokens from {cur_max_tokens} "
                            f"to {target_max_tokens} (attempt {truncate_attempt}/{context_truncate_max_retries}) and retrying."
                        )
                        continue

                new_messages, removed = _truncate_oldest_non_system_messages(
                    litellm_messages=litellm_messages,
                    drop_messages=context_truncate_drop_messages,
                )
                if removed <= 0:
                    logger.error(_format_exception(e))
                    raise e
                truncate_attempt += 1
                litellm_messages = new_messages
                logger.warning(
                    f"Context window exceeded; dropped {removed} oldest non-system messages "
                    f"(truncate attempt {truncate_attempt}/{context_truncate_max_retries}) and retrying."
                )
                continue
            logger.error(_format_exception(e))
            raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(_format_exception(e))
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
