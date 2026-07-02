from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4


def model_list_response(model_id: str) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


def chat_completion_response(
    *,
    model: str,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls:
        message["content"] = ""
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-local-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
    }


def streaming_chat_completion_events(response: dict[str, Any]) -> list[str]:
    chunk_base = {
        "id": _string_or_default(response.get("id"), f"chatcmpl-local-{uuid4().hex[:12]}"),
        "object": "chat.completion.chunk",
        "created": _int_or_default(response.get("created"), int(time.time())),
        "model": _string_or_default(response.get("model"), "qwen-agent"),
    }
    message = first_choice_message(response)
    finish_reason = _first_finish_reason(response)

    chunks = [
        _sse_data(
            {
                **chunk_base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )
    ]

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        chunks.append(
            _sse_data(
                {
                    **chunk_base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": _tool_call_deltas(tool_calls)},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        )
    else:
        content = message.get("content")
        if isinstance(content, str) and content:
            chunks.append(
                _sse_data(
                    {
                        **chunk_base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )

    chunks.append(
        _sse_data(
            {
                **chunk_base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
        )
    )
    chunks.append("data: [DONE]\n\n")
    return chunks


def first_choice_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return {"role": "assistant", "content": ""}
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return message if isinstance(message, dict) else {"role": "assistant", "content": ""}


def extract_content(response: dict[str, Any]) -> str:
    message = first_choice_message(response)
    content = message.get("content")
    return content if isinstance(content, str) else ""


def allowed_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def has_tool_results(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"tool", "function"}:
            return True
    return False


def _tool_call_deltas(tool_calls: list[Any]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = {}
        delta: dict[str, Any] = {
            "index": index,
            "id": _string_or_default(tool_call.get("id"), f"call_qwen_stream_{index + 1:04d}"),
            "type": "function",
            "function": {
                "name": _string_or_default(function.get("name"), ""),
                "arguments": _string_or_default(function.get("arguments"), ""),
            },
        }
        deltas.append(delta)
    return deltas


def _first_finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason
    return "stop"


def _sse_data(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _string_or_default(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _int_or_default(value: Any, default: int) -> int:
    return value if isinstance(value, int) else default
