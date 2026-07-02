from __future__ import annotations

import json
from typing import Any

import pytest

from qwen_agent_proxy.config import ComponentConfig, default_settings
from qwen_agent_proxy.orchestrator import QwenAgentOrchestrator, parse_planner_output


class FakeUpstream:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        component: ComponentConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "messages": messages,
                "component": component,
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return self.responses.pop(0)


def upstream_message(content: str = "", **extra: Any) -> dict[str, Any]:
    message = {"role": "assistant", "content": content}
    message.update(extra)
    return {"choices": [{"message": message}]}


def tool_schema(name: str = "read_file") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read a file.",
            "parameters": {"type": "object"},
        },
    }


def test_broken_planner_json_falls_back_safely() -> None:
    decision = parse_planner_output(
        {"role": "assistant", "content": "not json"},
        has_tools=True,
        has_tool_results=False,
    )
    assert decision.decision == "need_tool"


@pytest.mark.asyncio
async def test_no_tools_passes_through_as_normal_chat() -> None:
    settings = default_settings()
    upstream = FakeUpstream([upstream_message("<think>hidden</think>Hello")])
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert response["choices"][0]["message"]["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert upstream.calls[0]["tools"] is None


@pytest.mark.asyncio
async def test_no_tools_response_strips_raw_tool_call_markup() -> None:
    settings = default_settings()
    upstream = FakeUpstream(
        [
            upstream_message(
                '<tool_call>{"name":"read_file","arguments":{"path":"README.md"}}</tool_call>Done.'
            )
        ]
    )
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    message = response["choices"][0]["message"]
    assert message["content"] == "Done."
    assert "tool_calls" not in message


@pytest.mark.asyncio
async def test_tools_without_tool_results_return_tool_calls() -> None:
    planner = upstream_message(
        json.dumps(
            {
                "decision": "need_tool",
                "intent": "read_file",
                "rationale_summary": "Need workspace context.",
                "candidate_tool_names": ["read_file"],
                "tool_instruction": "Read src/main.ts.",
            }
        )
    )
    tool_caller = upstream_message(
        "",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "src/main.ts"},
                },
            }
        ],
    )
    settings = default_settings()
    upstream = FakeUpstream([planner, tool_caller])
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [tool_schema()],
        }
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == '{"path":"src/main.ts"}'
    assert upstream.calls[1]["tools"] == [tool_schema()]


@pytest.mark.asyncio
async def test_tool_result_history_goes_to_finalizer() -> None:
    planner = upstream_message(
        json.dumps(
            {
                "decision": "finalize",
                "intent": "answer",
                "rationale_summary": "Tool result is present.",
                "candidate_tool_names": [],
                "tool_instruction": "",
            }
        )
    )
    finalizer = upstream_message("<think>hidden</think>The file says hello.")
    settings = default_settings()
    upstream = FakeUpstream([planner, finalizer])
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [
                {"role": "user", "content": "read the file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"src/main.ts"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
            ],
            "tools": [tool_schema()],
        }
    )

    assert response["choices"][0]["message"]["content"] == "The file says hello."
    assert upstream.calls[1]["tools"] is None


@pytest.mark.asyncio
async def test_finalizer_repairable_tool_call_is_returned_as_tool_call() -> None:
    planner = upstream_message(
        json.dumps(
            {
                "decision": "answer_directly",
                "intent": "inspect_logs",
                "rationale_summary": "The model may still request a tool.",
                "candidate_tool_names": [],
                "tool_instruction": "",
            }
        )
    )
    finalizer = upstream_message(
        "<tool_call><function=run_in_terminal>"
        "<parameter command>echo hello</parameter>"
        "<parameter requireResult>true</parameter>"
        "</function></tool_call>"
    )
    settings = default_settings()
    upstream = FakeUpstream([planner, finalizer])
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "inspect logs"}],
            "tools": [tool_schema("run_in_terminal")],
        }
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "run_in_terminal"
    assert choice["message"]["tool_calls"][0]["function"]["arguments"] == (
        '{"command":"echo hello","requireResult":true}'
    )


@pytest.mark.asyncio
async def test_finalizer_unrepairable_tool_call_markup_does_not_leak() -> None:
    planner = upstream_message(
        json.dumps(
            {
                "decision": "answer_directly",
                "intent": "answer",
                "rationale_summary": "No allowed tool is needed.",
                "candidate_tool_names": [],
                "tool_instruction": "",
            }
        )
    )
    finalizer = upstream_message(
        '<tool_call>{"name":"unknown_tool","arguments":{"path":"README.md"}}</tool_call>Done.'
    )
    settings = default_settings()
    upstream = FakeUpstream([planner, finalizer])
    orchestrator = QwenAgentOrchestrator(settings, upstream)

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "answer directly"}],
            "tools": [tool_schema()],
        }
    )

    choice = response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "Done."
    assert "tool_calls" not in choice["message"]
