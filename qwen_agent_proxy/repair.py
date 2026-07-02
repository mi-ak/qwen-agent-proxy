from __future__ import annotations

import json
import re
from typing import Any

THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_CLOSE_PREFIX_RE = re.compile(r"^.*?</think>", re.IGNORECASE | re.DOTALL)
UNTERMINATED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
UNTERMINATED_TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
FUNCTION_TAG_RE = re.compile(r"<function=([A-Za-z_][A-Za-z0-9_.:-]*)\s*>(.*?)</function>", re.DOTALL)
PARAMETER_TAG_RE = re.compile(
    r"<parameter(?:\s+|=)([A-Za-z_][A-Za-z0-9_.:-]*)\s*>(.*?)</parameter>",
    re.DOTALL,
)
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


def strip_think(text: str) -> str:
    if not text:
        return ""
    stripped = THINK_BLOCK_RE.sub("", text)
    stripped = THINK_CLOSE_PREFIX_RE.sub("", stripped, count=1)
    stripped = UNTERMINATED_THINK_RE.sub("", stripped)
    return stripped.strip()


def strip_tool_call_markup(text: str) -> str:
    if not text:
        return ""
    stripped = TOOL_CALL_RE.sub("", text)
    stripped = UNTERMINATED_TOOL_CALL_RE.sub("", stripped)
    return stripped.strip()


def strip_client_visible_artifacts(text: str) -> str:
    return strip_tool_call_markup(strip_think(text))


def extract_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    if not isinstance(text, str) or not text:
        return []

    tool_calls: list[dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        body = match.group(1)
        parsed = _load_jsonish(body)
        if isinstance(parsed, dict):
            tool_calls.extend(_tool_call_dicts_from_object(parsed))
        elif isinstance(parsed, list):
            tool_calls.extend(item for item in parsed if isinstance(item, dict))
        else:
            tool_calls.extend(_parse_function_tag_tool_calls(body))
    if not tool_calls:
        tool_calls.extend(_parse_function_tag_tool_calls(text))
    return tool_calls


def extract_tool_calls_from_reasoning_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    reasoning_content = message.get("reasoning_content")
    if not isinstance(reasoning_content, str):
        return []
    return extract_xml_tool_calls(reasoning_content)


def ensure_json_string_arguments(arguments: object) -> str | None:
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

    if isinstance(arguments, str):
        candidate = _strip_code_fence(arguments).strip()
        if not candidate:
            return None
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return candidate

    return None


def normalize_tool_call(
    raw: dict[str, Any],
    allowed_tool_names: set[str],
) -> dict[str, Any] | None:
    function = raw.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    elif isinstance(raw.get("function_call"), dict):
        function_call = raw["function_call"]
        name = function_call.get("name")
        arguments = function_call.get("arguments")
    else:
        name = raw.get("name")
        arguments = raw.get("arguments")

    if not isinstance(name, str) or not name:
        return None
    if name not in allowed_tool_names:
        return None

    arguments_string = ensure_json_string_arguments(arguments)
    if arguments_string is None:
        return None

    tool_call_id = raw.get("id")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        tool_call_id = "call_qwen_repaired_0001"

    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_string,
        },
    }


def normalize_tool_calls(
    message: dict[str, Any],
    allowed_tool_names: set[str],
) -> list[dict[str, Any]]:
    raw_calls: list[dict[str, Any]] = []

    existing = message.get("tool_calls")
    if isinstance(existing, list):
        raw_calls.extend(item for item in existing if isinstance(item, dict))

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        raw_calls.append({"function_call": function_call})

    content = message.get("content")
    if isinstance(content, str):
        raw_calls.extend(extract_xml_tool_calls(content))

    raw_calls.extend(extract_tool_calls_from_reasoning_content(message))

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    repaired_index = 1
    for raw_call in raw_calls:
        tool_call = normalize_tool_call(raw_call, allowed_tool_names)
        if tool_call is None:
            continue
        original_id = raw_call.get("id")
        if not isinstance(original_id, str) or not original_id or tool_call["id"] in seen_ids:
            while True:
                candidate = f"call_qwen_repaired_{repaired_index:04d}"
                repaired_index += 1
                if candidate not in seen_ids:
                    tool_call["id"] = candidate
                    break
        seen_ids.add(tool_call["id"])
        normalized.append(tool_call)

    return normalized


def _strip_code_fence(text: str) -> str:
    match = FENCE_RE.match(text)
    return match.group(1) if match else text


def _load_jsonish(text: str) -> Any:
    candidate = _strip_code_fence(text).strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _tool_call_dicts_from_object(value: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list):
        return [item for item in tool_calls if isinstance(item, dict)]
    return [value]


def _parse_function_tag_tool_calls(text: str) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for function_match in FUNCTION_TAG_RE.finditer(text):
        name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments: dict[str, Any] = {}
        for parameter_match in PARAMETER_TAG_RE.finditer(body):
            parameter_name = parameter_match.group(1).strip()
            parameter_value = parameter_match.group(2).strip()
            arguments[parameter_name] = _coerce_parameter_value(parameter_value)
        tool_calls.append({"name": name, "arguments": arguments})
    return tool_calls


def _coerce_parameter_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    return value
