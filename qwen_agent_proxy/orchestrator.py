from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from qwen_agent_proxy.config import ComponentConfig, Settings
from qwen_agent_proxy.openai_types import (
    allowed_tool_names,
    chat_completion_response,
    extract_content,
    first_choice_message,
    has_tool_results,
)
from qwen_agent_proxy.prompts import finalizer_messages, planner_messages, tool_caller_messages
from qwen_agent_proxy.repair import (
    extract_tool_calls_from_reasoning_content,
    extract_xml_tool_calls,
    normalize_tool_calls,
    strip_client_visible_artifacts,
    strip_think,
)

LOGGER = logging.getLogger(__name__)

VALID_PLANNER_DECISIONS = {"answer_directly", "need_tool", "continue_tool", "finalize"}
FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)


class UpstreamClient(Protocol):
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        component: ComponentConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class PlannerDecision:
    decision: str
    intent: str = ""
    rationale_summary: str = ""
    candidate_tool_names: list[str] | None = None
    tool_instruction: str = ""

    @property
    def candidates(self) -> list[str]:
        return self.candidate_tool_names or []


class QwenAgentOrchestrator:
    def __init__(self, settings: Settings, upstream: UpstreamClient) -> None:
        self.settings = settings
        self.upstream = upstream

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = _as_message_list(request.get("messages"))
        tools = _as_tool_list(request.get("tools"))
        has_tools = bool(tools)
        has_results = has_tool_results(messages)

        if self.settings.agent.log_requests:
            LOGGER.info(
                "incoming request model=%s tools_count=%s has_tool_results=%s",
                request.get("model"),
                len(tools),
                has_results,
            )

        if not has_tools:
            upstream_response = await self.upstream.chat_completion(
                messages=messages,
                component=self.settings.finalizer,
            )
            content = strip_client_visible_artifacts(extract_content(upstream_response))
            return chat_completion_response(model=self.settings.agent.public_model_id, content=content)

        planner_decision = await self._plan(messages, tools, has_results)
        LOGGER.info(
            "planner decision=%s candidate_tools=%s",
            planner_decision.decision,
            planner_decision.candidates,
        )

        if has_results:
            if planner_decision.decision == "continue_tool":
                tool_response = await self._tool_call(messages, tools, planner_decision)
                if tool_response is not None:
                    return tool_response
            return await self._finalize(messages, planner_decision, tools)

        if planner_decision.decision in {"need_tool", "continue_tool"}:
            tool_response = await self._tool_call(messages, tools, planner_decision)
            if tool_response is not None:
                return tool_response
            return await self._finalize(messages, planner_decision, tools)

        return await self._finalize(messages, planner_decision, tools)

    async def _plan(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        has_results: bool,
    ) -> PlannerDecision:
        tool_names = sorted(allowed_tool_names(tools))
        upstream_response = await self.upstream.chat_completion(
            messages=planner_messages(messages, tool_names, has_results),
            component=self.settings.planner,
        )
        planner_message = first_choice_message(upstream_response)
        return parse_planner_output(
            planner_message,
            has_tools=bool(tools),
            has_tool_results=has_results,
        )

    async def _tool_call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        planner_decision: PlannerDecision,
    ) -> dict[str, Any] | None:
        allowed_names = allowed_tool_names(tools)
        max_retries = max(0, self.settings.agent.max_tool_retries)

        for retry_count in range(max_retries + 1):
            LOGGER.info("tool caller retry_count=%s", retry_count)
            upstream_response = await self.upstream.chat_completion(
                messages=tool_caller_messages(
                    messages,
                    planner_decision.tool_instruction,
                    retry_count,
                ),
                component=self.settings.tool_caller,
                tools=tools,
                tool_choice="auto",
            )
            tool_message = first_choice_message(upstream_response)
            rejected_tool_names = _rejected_tool_names(tool_message, allowed_names)
            tool_calls = normalize_tool_calls(tool_message, allowed_names)
            LOGGER.info(
                "repaired tool_calls count=%s rejected_tool_names=%s rejected_or_missing=%s",
                len(tool_calls),
                rejected_tool_names,
                not bool(tool_calls),
            )
            if tool_calls:
                return chat_completion_response(
                    model=self.settings.agent.public_model_id,
                    tool_calls=tool_calls,
                )

        return None

    async def _finalize(
        self,
        messages: list[dict[str, Any]],
        planner_decision: PlannerDecision,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        upstream_response = await self.upstream.chat_completion(
            messages=finalizer_messages(messages, planner_decision.rationale_summary),
            component=self.settings.finalizer,
        )
        message = first_choice_message(upstream_response)
        if tools:
            tool_calls = normalize_tool_calls(message, allowed_tool_names(tools))
            if tool_calls:
                LOGGER.info("finalizer emitted repairable tool_calls count=%s", len(tool_calls))
                return chat_completion_response(
                    model=self.settings.agent.public_model_id,
                    tool_calls=tool_calls,
                )

        content = strip_client_visible_artifacts(extract_content(upstream_response))
        return chat_completion_response(model=self.settings.agent.public_model_id, content=content)


def parse_planner_output(
    message: dict[str, Any],
    *,
    has_tools: bool,
    has_tool_results: bool,
) -> PlannerDecision:
    content = strip_think(message.get("content") if isinstance(message.get("content"), str) else "")
    parsed = _loads_json_object(content)
    if parsed is None:
        return fallback_planner_decision(has_tools, has_tool_results)

    decision = parsed.get("decision")
    if decision not in VALID_PLANNER_DECISIONS:
        return fallback_planner_decision(has_tools, has_tool_results)

    candidates = parsed.get("candidate_tool_names")
    if not isinstance(candidates, list):
        candidates = []
    candidates = [candidate for candidate in candidates if isinstance(candidate, str)]

    return PlannerDecision(
        decision=decision,
        intent=_string_value(parsed.get("intent")),
        rationale_summary=_string_value(parsed.get("rationale_summary")),
        candidate_tool_names=candidates,
        tool_instruction=_string_value(parsed.get("tool_instruction")),
    )


def fallback_planner_decision(has_tools: bool, has_tool_results: bool) -> PlannerDecision:
    if has_tool_results:
        decision = "finalize"
        instruction = "Use the existing tool results to produce a final answer."
    elif has_tools:
        decision = "need_tool"
        instruction = "Select the needed tool call using the available tools."
    else:
        decision = "answer_directly"
        instruction = "Answer directly without tools."
    return PlannerDecision(
        decision=decision,
        rationale_summary="Planner output was invalid; selected the safe fallback.",
        candidate_tool_names=[],
        tool_instruction=instruction,
    )


def _loads_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    match = FENCE_RE.match(candidate)
    if match:
        candidate = match.group(1).strip()
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _as_message_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_tool_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _rejected_tool_names(message: dict[str, Any], allowed_names: set[str]) -> list[str]:
    rejected = {
        name
        for name in _raw_tool_names(message)
        if isinstance(name, str) and name and name not in allowed_names
    }
    return sorted(rejected)


def _raw_tool_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    existing = message.get("tool_calls")
    if isinstance(existing, list):
        for raw_call in existing:
            names.extend(_name_from_raw_tool_call(raw_call))

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        name = function_call.get("name")
        if isinstance(name, str):
            names.append(name)

    content = message.get("content")
    if isinstance(content, str):
        for raw_call in extract_xml_tool_calls(content):
            names.extend(_name_from_raw_tool_call(raw_call))

    for raw_call in extract_tool_calls_from_reasoning_content(message):
        names.extend(_name_from_raw_tool_call(raw_call))

    return names


def _name_from_raw_tool_call(raw_call: Any) -> list[str]:
    if not isinstance(raw_call, dict):
        return []
    function = raw_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
    else:
        name = raw_call.get("name")
    return [name] if isinstance(name, str) else []
