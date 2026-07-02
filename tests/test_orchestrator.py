from __future__ import annotations

import asyncio
import json
import logging
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


class ComponentRoutingUpstream:
    def __init__(
        self,
        settings: Any,
        *,
        planner: dict[str, Any],
        tool_caller: dict[str, Any],
        finalizer: dict[str, Any] | None = None,
        planner_delay: float = 0.0,
    ) -> None:
        self.settings = settings
        self.planner = planner
        self.tool_caller = tool_caller
        self.finalizer = finalizer or upstream_message("done")
        self.planner_delay = planner_delay
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
        if component is self.settings.planner:
            if self.planner_delay:
                await asyncio.sleep(self.planner_delay)
            return self.planner
        if component is self.settings.tool_caller:
            return self.tool_caller
        if component is self.settings.finalizer:
            return self.finalizer
        raise AssertionError("unknown component")


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
async def test_tool_caller_warns_when_repair_finds_no_tool_calls(caplog: Any) -> None:
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
    settings = default_settings()
    settings.agent.max_tool_retries = 0
    upstream = FakeUpstream([planner, upstream_message(""), upstream_message("Need more info.")])
    orchestrator = QwenAgentOrchestrator(settings, upstream)
    caplog.set_level(logging.WARNING, logger="qwen_agent_proxy.orchestrator")

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [tool_schema()],
        }
    )

    assert response["choices"][0]["message"]["content"] == "Need more info."
    assert any(
        record.levelno == logging.WARNING
        and "tool caller repaired tool_calls count=0 retry_count=0 rejected_tool_names=[]"
        in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_tool_caller_warns_when_unknown_tool_names_are_rejected(caplog: Any) -> None:
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
                    "name": "delete_everything",
                    "arguments": {"path": "src/main.ts"},
                },
            }
        ],
    )
    settings = default_settings()
    settings.agent.max_tool_retries = 0
    upstream = FakeUpstream([planner, tool_caller, upstream_message("Need more info.")])
    orchestrator = QwenAgentOrchestrator(settings, upstream)
    caplog.set_level(logging.WARNING, logger="qwen_agent_proxy.orchestrator")

    response = await orchestrator.complete(
        {
            "model": "qwen-agent",
            "messages": [{"role": "user", "content": "read the file"}],
            "tools": [tool_schema()],
        }
    )

    assert response["choices"][0]["message"]["content"] == "Need more info."
    assert any(
        record.levelno == logging.WARNING
        and (
            "tool caller rejected unknown tool names retry_count=0 "
            "rejected_tool_names=['delete_everything']"
        )
        in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_parallel_tool_call_uses_finished_first_attempt_after_planner_allows_tools() -> None:
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
    settings.agent.parallel_tool_call = True
    upstream = ComponentRoutingUpstream(
        settings,
        planner=planner,
        tool_caller=tool_caller,
        planner_delay=0.01,
    )
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
    assert [call["component"] for call in upstream.calls].count(settings.tool_caller) == 1
    assert [call["component"] for call in upstream.calls].count(settings.planner) == 1
    tool_call = next(call for call in upstream.calls if call["component"] is settings.tool_caller)
    tool_payload = json.loads(tool_call["messages"][1]["content"])
    assert (
        tool_payload["planner_tool_instruction"]
        == "Select the needed tool call using the available tools."
    )


@pytest.mark.asyncio
async def test_parallel_tool_call_is_discarded_when_planner_answers_directly() -> None:
    planner = upstream_message(
        json.dumps(
            {
                "decision": "answer_directly",
                "intent": "answer",
                "rationale_summary": "No tool needed.",
                "candidate_tool_names": [],
                "tool_instruction": "",
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
    settings.agent.parallel_tool_call = True
    upstream = ComponentRoutingUpstream(
        settings,
        planner=planner,
        tool_caller=tool_caller,
        finalizer=upstream_message("Answered without tools."),
        planner_delay=0.01,
    )
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
    assert choice["message"]["content"] == "Answered without tools."
    assert "tool_calls" not in choice["message"]
    assert [call["component"] for call in upstream.calls].count(settings.tool_caller) == 1
    assert [call["component"] for call in upstream.calls].count(settings.finalizer) == 1


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
