# AI Bot Guide

This file is for AI coding agents and automation bots working on
`qwen-agent-proxy`. Human setup and usage instructions live in `README.md`.

## Project Intent

`qwen-agent-proxy` exposes one OpenAI Chat Completions compatible model to
VS Code BYOK / Copilot Agent and internally separates Qwen reasoning from
tool emission.

Preserve this invariant:

```text
Reasoning components must not receive tools.
Tool-calling components must not reason.
The client must only see valid OpenAI-compatible tool_calls.
```

## Compatibility Contract

The public API is:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

The public model ID defaults to `qwen-agent`.

When returning tool calls, the response must use:

- `choices[0].message.tool_calls`
- `choices[0].message.content == ""`
- `choices[0].finish_reason == "tool_calls"`
- `function.arguments` as a JSON string, not a dict

Do not leak these to the client:

- `<think>...</think>`
- `reasoning_content`
- raw `<tool_call>...</tool_call>` XML
- malformed tool-call JSON
- unknown tool names

## Request Flow

No tools in client request:

```text
client -> upstream normal chat -> strip <think> -> client
```

Tools present, no tool results:

```text
client -> Planner(reasoning on, no tools)
       -> Tool Caller(reasoning off, tools)
       -> repair/normalize tool_calls
       -> client
```

Tool results present:

```text
client -> Planner(reasoning on, no tools)
       -> Finalizer(reasoning on, no tools)
       -> strip <think>
       -> client
```

If Planner output is invalid, use the safe fallback in
`orchestrator.fallback_planner_decision`.

## File Map

- `qwen_agent_proxy/main.py`: FastAPI app and route handlers.
- `qwen_agent_proxy/config.py`: default settings and TOML loading.
- `qwen_agent_proxy/openai_types.py`: OpenAI-compatible response helpers.
- `qwen_agent_proxy/upstream.py`: `httpx.AsyncClient` upstream calls.
- `qwen_agent_proxy/orchestrator.py`: Planner / Tool Caller / Finalizer flow.
- `qwen_agent_proxy/repair.py`: `<think>` stripping and tool-call normalization.
- `qwen_agent_proxy/prompts.py`: internal component prompts.
- `qwen_agent_proxy/logging_utils.py`: logging helpers and redaction.
- `tests/`: pytest coverage for repair, orchestration, and API compatibility.

## Change Rules

- Keep upstream calls non-streaming unless the task explicitly implements true
  streaming. Client `stream: true` is handled by the SSE fallback in `main.py`.
- Do not pass `tools` to Planner or Finalizer.
- Do not enable thinking for Tool Caller.
- Do not silently turn broken arguments into `{}`.
- Do not accept tool names outside `tools[*].function.name`.
- Do not log API keys or full authorization headers.
- Prefer small deterministic helpers over model-dependent repair behavior.
- Add or update tests for every behavior change in orchestration or repair.

## Common Tasks

Add support for a new upstream thinking parameter:

1. Update `UpstreamConfig.thinking_param_style` handling in `upstream.py`.
2. Add the documented value to `config.example.toml`.
3. Mention it in `README.md`.
4. Add a focused unit test if request shaping becomes non-trivial.

Change tool-call repair:

1. Update `repair.py`.
2. Add examples to `tests/test_repair.py`.
3. Check that unknown names and invalid arguments are still rejected.

Change request routing:

1. Update `orchestrator.py`.
2. Add fake-upstream coverage in `tests/test_orchestrator.py`.
3. Confirm no path leaks `<think>` content.

## Verification

Run:

```bash
uv run pytest
uv run ruff check
```

For endpoint smoke tests:

```bash
uv run uvicorn qwen_agent_proxy.main:app --host 127.0.0.1 --port 9011
curl http://127.0.0.1:9011/health
curl http://127.0.0.1:9011/v1/models
```

## Research Notes

The interesting claim is not "repair malformed XML". The stronger claim is:

```text
Protocol-safe agent behavior improves when reasoning and action emission are
separated into different inference roles.
```

If adding benchmarks, compare at least:

- raw reasoning model with tools
- no-thinking tool caller only
- repair-only proxy
- role-separated Planner / Tool Caller / Finalizer
