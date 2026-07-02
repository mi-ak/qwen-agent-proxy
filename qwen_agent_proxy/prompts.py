from __future__ import annotations

import json
from typing import Any


PLANNER_PROMPT = """You are the planning component of a coding agent.

You may reason deeply, but you must not call tools.
You will receive the user request, conversation history, and a list of available tool names.
Decide whether a tool is needed.

Return only JSON with:
- decision
- intent
- rationale_summary
- candidate_tool_names
- tool_instruction

Valid decision values:
- answer_directly
- need_tool
- continue_tool
- finalize

Do not include tool calls.
Do not include XML.
Do not answer the user directly.
"""


TOOL_CALLER_PROMPT = """You are the tool-calling component of a coding agent.

Your only job is to emit valid OpenAI-compatible tool_calls.
Do not explain.
Do not answer the user.
Do not include reasoning.
Do not include markdown.
Do not include <think>.
Do not include <tool_call> in content.

Use only the tools provided.
If a tool is needed, emit tool_calls.
If no tool is needed, return an empty final response.
"""


FINALIZER_PROMPT = """You are the final response component of a coding agent.

Use the conversation and tool results to answer the user.
Do not claim that you used a tool unless a tool result is present.
Do not expose hidden reasoning.
Do not include raw tool call JSON.
If more workspace information is required, say that another tool call is needed.
"""


def planner_messages(
    messages: list[dict[str, Any]],
    available_tool_names: list[str],
    has_tool_results: bool,
) -> list[dict[str, str]]:
    payload = {
        "conversation_messages": messages,
        "available_tool_names": available_tool_names,
        "has_tool_results": has_tool_results,
    }
    return [
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def tool_caller_messages(
    messages: list[dict[str, Any]],
    planner_instruction: str,
    retry_count: int,
) -> list[dict[str, str]]:
    payload = {
        "conversation_messages": messages,
        "planner_tool_instruction": planner_instruction,
        "retry_count": retry_count,
    }
    if retry_count > 0:
        payload["retry_note"] = (
            "The previous output did not contain valid OpenAI-compatible tool_calls. "
            "Return only valid tool_calls using the supplied tools."
        )
    return [
        {"role": "system", "content": TOOL_CALLER_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def finalizer_messages(
    messages: list[dict[str, Any]],
    planner_summary: str | None = None,
) -> list[dict[str, str]]:
    final_messages = [{"role": "system", "content": FINALIZER_PROMPT}]
    if planner_summary:
        final_messages.append(
            {
                "role": "system",
                "content": f"Planner summary for this turn: {planner_summary}",
            }
        )
    final_messages.extend(messages)
    return final_messages
