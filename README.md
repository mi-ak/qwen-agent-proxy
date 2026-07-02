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
OpenAI-compatible Qwen API
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
[server]
host = "127.0.0.1"
port = 9011

[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen3.6-27B-MTPLX-Optimized-Speed"
thinking_param_style = "extra_body"

[agent]
public_model_id = "qwen-agent"
max_tool_retries = 2
streaming = false
log_requests = true
log_upstream = true

[planner]
enable_thinking = true
temperature = 0.2
max_tokens = 4096

[tool_caller]
enable_thinking = false
temperature = 0.0
max_tokens = 2048

[finalizer]
enable_thinking = true
temperature = 0.3
max_tokens = 8192

[repair]
extract_xml_tool_call = true
extract_reasoning_content_tool_call = true
json_repair = true
validate_tool_name = true
strip_think = true
```

`thinking_param_style` supports:

- `extra_body`: sends `{"extra_body":{"enable_thinking":true}}`
- `top_level`: sends `{"enable_thinking":true}`
- `disabled`: sends no thinking parameter

## Run

```bash
uv run uvicorn qwen_agent_proxy.main:app --host 127.0.0.1 --port 9011
```

## Upstream Examples

The upstream must expose an OpenAI-compatible `/v1/chat/completions` endpoint.

OMLX / MLX-style local server:

```toml
[upstream]
base_url = "http://127.0.0.1:8000/v1"
api_key = "dummy"
model = "Qwen3.6-27B-MTPLX-Optimized-Speed"
thinking_param_style = "extra_body"
```

llama.cpp server:

```bash
llama-server --host 127.0.0.1 --port 8000 --jinja -hf Qwen/Qwen3-8B-GGUF
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
thinking_param_style = "top_level"
```

If an upstream rejects unknown fields, set `thinking_param_style = "disabled"`.

## VS Code BYOK

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

The proxy normalizes these forms into OpenAI-compatible `tool_calls`:

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
the names supplied in `tools[*].function.name`.

## Logs

The service logs:

- incoming request model
- tools count
- whether tool results are present
- planner decision and candidate tools
- tool caller retry count
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
- Check that `[repair].strip_think = true`.

## License

MIT. See [LICENSE](LICENSE).
