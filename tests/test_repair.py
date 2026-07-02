from __future__ import annotations

import json

from qwen_agent_proxy.repair import (
    ensure_json_string_arguments,
    extract_tool_calls_from_reasoning_content,
    extract_xml_tool_calls,
    normalize_tool_call,
    normalize_tool_calls,
    strip_client_visible_artifacts,
    strip_think,
)


def test_strip_think_removes_single_block() -> None:
    assert strip_think("<think>I should reason.</think>Hello") == "Hello"


def test_strip_think_removes_multiple_blocks() -> None:
    text = "<think>one</think>Hello<think>two</think> world"
    assert strip_think(text) == "Hello world"


def test_strip_think_removes_qwen_closing_tag_prefix() -> None:
    text = "I should reason privately.</think>\nVisible answer."
    assert strip_think(text) == "Visible answer."


def test_strip_think_removes_unclosed_block() -> None:
    text = "Visible prefix. <think>unfinished hidden reasoning"
    assert strip_think(text) == "Visible prefix."


def test_strip_client_visible_artifacts_removes_raw_tool_call_markup() -> None:
    text = '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>Done.'
    assert strip_client_visible_artifacts(text) == "Done."


def test_strip_client_visible_artifacts_removes_unclosed_tool_call_markup() -> None:
    text = 'Visible prefix. <tool_call>{"name":"read_file"'
    assert strip_client_visible_artifacts(text) == "Visible prefix."


def test_extract_xml_tool_calls() -> None:
    text = """
<think>I need to read the file.</think>
<tool_call>
{"name":"read_file","arguments":{"path":"src/main.ts"}}
</tool_call>
"""
    calls = extract_xml_tool_calls(text)
    assert calls == [{"name": "read_file", "arguments": {"path": "src/main.ts"}}]


def test_extract_multiple_xml_tool_calls() -> None:
    text = """
<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>
<tool_call>{"name":"list_dir","arguments":{"path":"tests"}}</tool_call>
"""
    calls = extract_xml_tool_calls(text)
    assert calls == [
        {"name": "read_file", "arguments": {"path": "README.md"}},
        {"name": "list_dir", "arguments": {"path": "tests"}},
    ]


def test_extract_function_tag_tool_call() -> None:
    text = r"""
<tool_call>
<function=run_in_terminal>
<parameter command>
ls -lh ./fixtures/Application\ Support/logs/
</parameter>
<parameter requireResult>
true
</parameter>
</function>
</tool_call>
"""
    calls = extract_xml_tool_calls(text)
    assert calls == [
        {
            "name": "run_in_terminal",
            "arguments": {
                "command": (
                    r"ls -lh ./fixtures/Application\ "
                    r"Support/logs/"
                ),
                "requireResult": True,
            },
        }
    ]


def test_extract_unclosed_tool_call_with_equals_parameter() -> None:
    text = (
        "<tool_call> <function=run_in_terminal> <parameter command> "
        'LOGDIR="./fixtures/Application Support/logs" '
        '&& ls -lh "$LOGDIR/" '
        "</parameter> <parameter=requireResult> true </parameter> </function>"
    )
    calls = extract_xml_tool_calls(text)
    assert calls == [
        {
            "name": "run_in_terminal",
            "arguments": {
                "command": (
                    'LOGDIR="./fixtures/Application Support/logs" && ls -lh "$LOGDIR/"'
                ),
                "requireResult": True,
            },
        }
    ]


def test_extract_tool_calls_from_reasoning_content() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "reasoning_content": (
            '<tool_call>{"name":"read_file","arguments":{"path":"src/main.ts"}}</tool_call>'
        ),
    }
    calls = extract_tool_calls_from_reasoning_content(message)
    assert calls == [{"name": "read_file", "arguments": {"path": "src/main.ts"}}]


def test_arguments_dict_becomes_json_string() -> None:
    result = ensure_json_string_arguments({"path": "src/main.ts"})
    assert result == '{"path":"src/main.ts"}'


def test_arguments_list_becomes_json_string() -> None:
    result = ensure_json_string_arguments(["README.md", "tests"])
    assert result == '["README.md","tests"]'


def test_json_string_arguments_are_preserved() -> None:
    result = ensure_json_string_arguments('{"path":"README.md"}')
    assert result == '{"path":"README.md"}'


def test_invalid_json_string_arguments_reject_tool_call() -> None:
    raw = {"name": "read_file", "arguments": "{path: src/main.ts}"}
    assert normalize_tool_call(raw, {"read_file"}) is None


def test_unknown_tool_name_is_rejected() -> None:
    raw = {"name": "delete_everything", "arguments": "{}"}
    assert normalize_tool_call(raw, {"read_file"}) is None


def test_openai_compatible_tool_calls_are_normalized() -> None:
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "src/main.ts"},
                },
            }
        ],
    }
    calls = normalize_tool_calls(message, {"read_file"})
    assert calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"src/main.ts"}',
            },
        }
    ]
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "src/main.ts"}


def test_function_tag_tool_call_is_normalized() -> None:
    message = {
        "role": "assistant",
        "content": (
            "<tool_call><function=run_in_terminal>"
            "<parameter command>echo hello</parameter>"
            "<parameter requireResult>true</parameter>"
            "</function></tool_call>"
        ),
    }
    calls = normalize_tool_calls(message, {"run_in_terminal"})
    assert calls == [
        {
            "id": "call_qwen_repaired_0001",
            "type": "function",
            "function": {
                "name": "run_in_terminal",
                "arguments": '{"command":"echo hello","requireResult":true}',
            },
        }
    ]
