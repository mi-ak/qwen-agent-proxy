from __future__ import annotations

import logging
from typing import Any

import httpx

from qwen_agent_proxy.config import ComponentConfig, UpstreamConfig
from qwen_agent_proxy.logging_utils import redact_headers

LOGGER = logging.getLogger(__name__)
DEFAULT_PROVIDER_NAME = "default"


class UpstreamError(RuntimeError):
    pass


class OpenAICompatibleUpstream:
    def __init__(
        self,
        config: UpstreamConfig,
        log_upstream: bool = True,
        providers: dict[str, UpstreamConfig] | None = None,
    ) -> None:
        self.config = config
        self.log_upstream = log_upstream
        self.providers = providers or {}

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        component: ComponentConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider_name, provider, url, headers, payload = self._build_request(
            messages=messages,
            component=component,
            tools=tools,
            tool_choice=tool_choice,
        )

        if self.log_upstream:
            LOGGER.info(
                (
                    "upstream request provider=%s url=%s model=%s messages=%s tools=%s "
                    "thinking_style=%s headers=%s"
                ),
                provider_name,
                url,
                payload["model"],
                len(messages),
                len(tools or []),
                provider.thinking_param_style,
                redact_headers(headers),
            )

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            LOGGER.exception("upstream error type=%s", exc.__class__.__name__)
            raise UpstreamError(f"upstream request failed: {exc}") from exc

        if self.log_upstream:
            LOGGER.info("upstream status=%s", response.status_code)

        if response.status_code >= 400:
            raise UpstreamError(f"upstream returned HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError as exc:
            raise UpstreamError("upstream returned non-JSON response") from exc

        if not isinstance(data, dict):
            raise UpstreamError("upstream returned invalid response shape")
        return data

    def _build_request(
        self,
        *,
        messages: list[dict[str, Any]],
        component: ComponentConfig,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> tuple[str, UpstreamConfig, str, dict[str, str], dict[str, Any]]:
        provider_name, provider = self._resolve_provider(component.provider)
        url = f"{provider.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": component.model or provider.model,
            "messages": messages,
            "temperature": component.temperature,
            "max_tokens": component.max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        self._apply_thinking_param(payload, component.enable_thinking, provider)
        return provider_name, provider, url, headers, payload

    def _resolve_provider(self, provider_name: str | None) -> tuple[str, UpstreamConfig]:
        name = provider_name or DEFAULT_PROVIDER_NAME
        if name == DEFAULT_PROVIDER_NAME:
            return DEFAULT_PROVIDER_NAME, self.config
        provider = self.providers.get(name)
        if provider is None:
            raise UpstreamError(f"unknown upstream provider: {name}")
        return name, provider

    def _apply_thinking_param(
        self,
        payload: dict[str, Any],
        enable_thinking: bool,
        provider: UpstreamConfig | None = None,
    ) -> None:
        style = (provider or self.config).thinking_param_style
        if style == "chat_template_kwargs":
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        elif style == "extra_body_chat_template_kwargs":
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        elif style == "extra_body":
            payload["extra_body"] = {"enable_thinking": enable_thinking}
        elif style == "top_level":
            payload["enable_thinking"] = enable_thinking
        elif style == "disabled":
            return
        else:
            LOGGER.warning("unknown thinking_param_style=%s; not sending thinking parameter", style)
