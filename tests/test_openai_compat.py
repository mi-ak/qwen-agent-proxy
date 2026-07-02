from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from qwen_agent_proxy.openai_types import chat_completion_response, streaming_chat_completion_events
from qwen_agent_proxy.main import app


def test_models_returns_qwen_agent() -> None:
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "qwen-agent",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


def test_chat_completions_rejects_invalid_json() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "request body must be valid JSON"


def test_chat_completions_rejects_non_object_json() -> None:
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json=[])

    assert response.status_code == 400
    assert response.json()["detail"] == "request body must be a JSON object"


def test_streaming_content_events_from_non_streaming_response() -> None:
    response = chat_completion_response(model="qwen-agent", content="hello")
    events = streaming_chat_completion_events(response)

    assert events[0].startswith("data: ")
    assert '"object":"chat.completion.chunk"' in events[0]
    assert '"role":"assistant"' in events[0]
    assert '"content":"hello"' in "".join(events)
    assert '"finish_reason":"stop"' in "".join(events)
    assert events[-1] == "data: [DONE]\n\n"


def test_streaming_tool_call_events_from_non_streaming_response() -> None:
    response = chat_completion_response(
        model="qwen-agent",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                },
            }
        ],
    )
    events = streaming_chat_completion_events(response)

    joined = "".join(events)
    assert '"tool_calls"' in joined
    assert '"name":"read_file"' in joined
    assert '"finish_reason":"tool_calls"' in joined
    assert events[-1] == "data: [DONE]\n\n"


def test_streaming_request_uses_sse_fallback(monkeypatch: Any) -> None:
    class FakeOrchestrator:
        async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
            assert payload["stream"] is False
            return chat_completion_response(model="qwen-agent", content="hello")

    import qwen_agent_proxy.main as main

    monkeypatch.setattr(main, "orchestrator", FakeOrchestrator())
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen-agent",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]" in response.text
    content_events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]
    parsed_events = [json.loads(event) for event in content_events]
    assert any(
        choice["delta"].get("content") == "hello"
        for event in parsed_events
        for choice in event["choices"]
    )
