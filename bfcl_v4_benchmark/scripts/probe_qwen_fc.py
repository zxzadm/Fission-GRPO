import json
import os

from openai import OpenAI


BASE_URL = os.getenv("MINING_BASE_URL", "")
API_KEY = os.getenv("MINING_API_KEY", "")
MODEL = os.getenv("QWEN_MODEL", "qwen3.5-397b-a17b")


def main():
    if not API_KEY:
        raise RuntimeError("Missing MINING_API_KEY")
    if not BASE_URL:
        raise RuntimeError("Missing MINING_BASE_URL")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two integers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    print("=== STEP1: baseline ===")
    r1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "1+1=?"},
        ],
        stream=False,
        max_tokens=256,
        temperature=0,
    )
    print("finish_reason:", r1.choices[0].finish_reason)
    print("content:", r1.choices[0].message.content)

    print("\n=== STEP2: with tools (no tool_choice) ===")
    r2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "When tools are available, prefer calling tools over direct answers.",
            },
            {"role": "user", "content": "Use add to compute 1+1."},
        ],
        tools=tools,
        stream=False,
        max_tokens=256,
        temperature=0,
    )
    msg2 = r2.choices[0].message
    print("finish_reason:", r2.choices[0].finish_reason)
    print("tool_calls:", json.dumps(msg2.tool_calls, ensure_ascii=False, default=str))
    print("content:", msg2.content)

    print("\n=== STEP3: with tool_choice ===")
    r3 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Call the tool."},
            {"role": "user", "content": "Compute 1+1."},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "add"}},
        stream=False,
        max_tokens=256,
        temperature=0,
    )
    msg3 = r3.choices[0].message
    print("finish_reason:", r3.choices[0].finish_reason)
    print("tool_calls:", json.dumps(msg3.tool_calls, ensure_ascii=False, default=str))
    print("content:", msg3.content)

    print("\n=== SUMMARY ===")
    if msg2.tool_calls:
        print("FC usable via tools.")
    else:
        print("tools accepted, but no tool_calls emitted.")
    if msg3.tool_calls:
        print("tool_choice appears accepted.")
    else:
        print("tool_choice appears unsupported or ignored.")


if __name__ == "__main__":
    main()
