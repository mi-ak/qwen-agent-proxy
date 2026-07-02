# qwen-agent-proxy

`qwen-agent-proxy` is a small FastAPI service that presents one
OpenAI Chat Completions compatible model to VS Code BYOK / Copilot Agent
while internally separating Qwen reasoning from Qwen tool calling.

License: MIT.

Documentation:

- Human users: start here.
- AI coding agents and automation bots: read [AGENTS.md](AGENTS.md) before changing code.

The core rule is:

```text
Do not give tools to the reasoning component.
Do not ask the tool-calling component to reason.
Return only valid OpenAI-compatible tool_calls to VS Code.
```

## Why

Qwen reasoning / thinking mode can expose tool-call shaped text in places that
agent clients do not expect:

- inside `<think>...</think>`
- inside `reasoning_content`
- inside plain content as `<tool_call>...</tool_call>`

VS Code BYOK / Copilot Agent expects standard OpenAI-compatible
`message.tool_calls` with `finish_reason: "tool_calls"`. This proxy avoids the
problem structurally by splitting each turn into focused internal components.

## Architecture

```text
VS Code BYOK / Copilot Agent
  |
  v
qwen-agent-proxy
  |-- Planner      : reasoning on,  no tools
  |-- Tool Caller  : reasoning off, tools allowed
  `-- Finalizer    : reasoning on,  no tools
  |
  v
OpenAI-compatible upstream provider(s)
```

Planner decides whether a tool is needed. Tool Caller emits only tool calls.
Finalizer turns tool results into the final natural-language answer.

## Install

```bash
uv sync
```

Or install only runtime dependencies:

```bash
uv sync --no-dev
```

## Configure

Copy the example file:

```bash
cp config.example.toml config.toml
```

Config loading order:

1. `QWEN_AGENT_PROXY_CONFIG`
2. `./config.toml`
3. built-in defaults

Example:

```toml
[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen/Qwen3-8B"
thinking_param_style = "chat_template_kwargs"

# Optional named providers inherit missing values from [upstream].
# [providers.fast_tools]
# base_url = "http://127.0.0.1:8001/v1"
# api_key = "dummy"
# model = "Qwen/Qwen3-4B"
# thinking_param_style = "disabled"

[agent]
public_model_id = "qwen-agent"
max_tool_retries = 2
parallel_tool_call = false
log_requests = true
log_upstream = true

[planner]
# provider = "default"
# model = "Qwen/Qwen3-14B"
enable_thinking = true
temperature = 0.2
max_tokens = 4096

[tool_caller]
# provider = "fast_tools"
# model = "Qwen/Qwen3-4B"
enable_thinking = false
temperature = 0.0
max_tokens = 2048

[finalizer]
# provider = "default"
# model = "Qwen/Qwen3-14B"
enable_thinking = true
temperature = 0.3
max_tokens = 8192
```

`[upstream]` is always the `default` provider. Add `[providers.<name>]` blocks
when a component should use a different OpenAI-compatible endpoint, API key,
thinking parameter style, or default model. Named providers inherit missing
values from `[upstream]`.

Each internal component can set:

- `provider`: provider name, defaulting to `default`
- `model`: optional per-component model override; if omitted, the provider's
  model is used

`parallel_tool_call = true` starts a speculative first Tool Caller attempt while
Planner is still running for tool-enabled requests without tool results. The
proxy only returns that tool call if Planner also decides a tool is needed; if
Planner finishes first or decides to answer directly, the speculative result is
discarded. This can reduce latency when Planner uses a slower reasoning model,
at the cost of extra upstream work.

`thinking_param_style` supports:

- `chat_template_kwargs`: sends `{"chat_template_kwargs":{"enable_thinking":true}}`
- `extra_body_chat_template_kwargs`: sends `{"extra_body":{"chat_template_kwargs":{"enable_thinking":true}}}`
- `extra_body`: legacy style; sends `{"extra_body":{"enable_thinking":true}}`
- `top_level`: legacy style; sends `{"enable_thinking":true}`
- `disabled`: sends no thinking parameter

## Run

```bash
uv run uvicorn qwen_agent_proxy.main:app --host 127.0.0.1 --port 9011
```

## Upstream Examples

The upstream must expose an OpenAI-compatible `/v1/chat/completions` endpoint.

vLLM with current Qwen chat templates:

```bash
vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-reasoning \
  --reasoning-parser qwen3
```

```toml
[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen/Qwen3-8B"
thinking_param_style = "chat_template_kwargs"
```

llama.cpp server:

```bash
llama-server --host 127.0.0.1 --port 8000 \
  --jinja \
  --reasoning-format deepseek \
  -hf Qwen/Qwen3-8B-GGUF
```

```toml
[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen3-8B"
thinking_param_style = "disabled"
```

LM Studio:

```toml
[upstream]
base_url = "http://127.0.0.1:1234/v1"
api_key = "dummy"
model = "qwen3-local"
thinking_param_style = "disabled"
```

Some local servers expose their own non-standard thinking switch. If an upstream
rejects unknown fields, set `thinking_param_style = "disabled"` or use the
server's non-thinking chat template for Tool Caller traffic.

You can also split components across providers:

```toml
[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen/Qwen3-14B"
thinking_param_style = "chat_template_kwargs"

[providers.fast_tools]
base_url = "http://127.0.0.1:8001/v1"
model = "Qwen/Qwen3-4B"
thinking_param_style = "disabled"

[tool_caller]
provider = "fast_tools"
enable_thinking = false
temperature = 0.0
max_tokens = 2048
```

## VS Code BYOK

VS Code's Custom Endpoint provider supports Chat Completions, Responses, and
Messages API endpoints. Configure this proxy as a Chat Completions endpoint and
keep `toolCalling` enabled; VS Code only shows models for agent use when the
model declares tool-calling support.

Example `chatLanguageModels.json`:

```json
[
  {
    "name": "Local Qwen Agent",
    "vendor": "customendpoint",
    "apiKey": "dummy",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "qwen-agent",
        "name": "Qwen Agent",
        "url": "http://127.0.0.1:9011/v1/chat/completions",
        "toolCalling": true,
        "vision": false,
        "maxInputTokens": 262144,
        "maxOutputTokens": 16384
      }
    ]
  }
]
```

## Smoke Tests

```bash
curl http://127.0.0.1:9011/health
curl http://127.0.0.1:9011/v1/models
```

Chat:

```bash
curl http://127.0.0.1:9011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-agent",
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

Tool-call shape:

```bash
curl http://127.0.0.1:9011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen-agent",
    "messages": [{"role": "user", "content": "Read src/main.ts"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "read_file",
          "description": "Read a file",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
          }
        }
      }
    ]
  }'
```

## Repair Behavior

The proxy always strips Qwen thinking content from client-visible messages and
normalizes these forms into OpenAI-compatible `tool_calls`:

```xml
<tool_call>
{"name":"read_file","arguments":{"path":"src/main.ts"}}
</tool_call>
```

```json
{
  "role": "assistant",
  "content": "",
  "reasoning_content": "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/main.ts\"}}</tool_call>"
}
```

Invalid JSON arguments are rejected. Unknown tool names are rejected based on
the names supplied in `tools[*].function.name`. These safety checks are not
configurable because they are part of the client compatibility contract.

## Logs

The service logs:

- incoming request model
- tools count
- whether tool results are present
- planner decision and candidate tools
- tool caller retry count
- upstream provider and model
- repaired tool call count
- upstream status or error

Authorization headers are redacted.

## Tests

```bash
uv run pytest
uv run ruff check
```

## v0.1 Limits

- Upstream calls are non-streaming. Client requests with `"stream": true` get
  a compatibility SSE stream generated from the completed non-streaming result.
- No heavy JSON repair library is used.
- Only OpenAI-compatible function tools are supported.
- The service exposes one public model ID.
- Vision and multimodal inputs are not implemented.

## Troubleshooting

If VS Code does not call tools:

- Check that `/v1/models` returns `qwen-agent`.
- Check that the VS Code config has `"toolCalling": true`.
- Check logs for `tools_count=0`.
- Verify that upstream accepts the configured `thinking_param_style`.

If tool calls are missing:

- Check logs for `repaired tool_calls count=0`.
- Confirm the upstream can emit OpenAI-compatible tool calls when thinking is off.
- Confirm tool names exactly match `tools[*].function.name`.

If `<think>` leaks:

- Ensure traffic goes through this proxy, not directly to the upstream server.
- Capture the upstream response and add a focused `strip_think` regression test.

## License

MIT. See [LICENSE](LICENSE).
