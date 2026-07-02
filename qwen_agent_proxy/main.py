from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from qwen_agent_proxy.config import load_config
from qwen_agent_proxy.logging_utils import configure_logging
from qwen_agent_proxy.openai_types import model_list_response, streaming_chat_completion_events
from qwen_agent_proxy.orchestrator import QwenAgentOrchestrator
from qwen_agent_proxy.upstream import OpenAICompatibleUpstream, UpstreamError

configure_logging()

settings = load_config()
upstream = OpenAICompatibleUpstream(settings.upstream, log_upstream=settings.agent.log_upstream)
orchestrator = QwenAgentOrchestrator(settings, upstream)
app = FastAPI(title="qwen-agent-proxy", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return model_list_response(settings.agent.public_model_id)


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> dict[str, Any] | StreamingResponse:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    client_requested_stream = payload.get("stream") is True
    if client_requested_stream:
        payload = {**payload, "stream": False}

    try:
        response = await orchestrator.complete(payload)
    except UpstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if client_requested_stream:
        return StreamingResponse(
            iter(streaming_chat_completion_events(response)),
            media_type="text/event-stream",
        )
    return response
