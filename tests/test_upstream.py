from __future__ import annotations

import pytest

from qwen_agent_proxy.config import ComponentConfig, UpstreamConfig, default_settings
from qwen_agent_proxy.upstream import OpenAICompatibleUpstream, UpstreamError


def apply_thinking_param(style: str, enable_thinking: bool = True) -> dict[str, object]:
    payload: dict[str, object] = {}
    upstream = OpenAICompatibleUpstream(UpstreamConfig(thinking_param_style=style))
    upstream._apply_thinking_param(payload, enable_thinking)
    return payload


def test_default_thinking_param_style_uses_qwen_vllm_shape() -> None:
    assert default_settings().upstream.thinking_param_style == "chat_template_kwargs"


def test_chat_template_kwargs_thinking_param_shape() -> None:
    assert apply_thinking_param("chat_template_kwargs", False) == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_extra_body_chat_template_kwargs_thinking_param_shape() -> None:
    assert apply_thinking_param("extra_body_chat_template_kwargs") == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
    }


def test_legacy_thinking_param_shapes() -> None:
    assert apply_thinking_param("extra_body") == {"extra_body": {"enable_thinking": True}}
    assert apply_thinking_param("top_level") == {"enable_thinking": True}


def test_disabled_thinking_param_shape() -> None:
    assert apply_thinking_param("disabled") == {}


def test_component_provider_and_model_override_shape_request() -> None:
    upstream = OpenAICompatibleUpstream(
        UpstreamConfig(
            base_url="http://default.example/v1",
            api_key="default-key",
            model="default-model",
            thinking_param_style="chat_template_kwargs",
        ),
        providers={
            "tools": UpstreamConfig(
                base_url="http://tools.example/v1",
                api_key="tools-key",
                model="tools-default-model",
                thinking_param_style="disabled",
            )
        },
    )

    provider_name, provider, url, headers, payload = upstream._build_request(
        messages=[{"role": "user", "content": "hello"}],
        component=ComponentConfig(
            provider="tools",
            model="tools-override-model",
            enable_thinking=False,
            temperature=0.0,
            max_tokens=128,
        ),
        tools=[{"type": "function", "function": {"name": "read_file"}}],
        tool_choice="auto",
    )

    assert provider_name == "tools"
    assert provider.model == "tools-default-model"
    assert url == "http://tools.example/v1/chat/completions"
    assert headers["Authorization"] == "Bearer tools-key"
    assert payload["model"] == "tools-override-model"
    assert payload["tool_choice"] == "auto"
    assert "chat_template_kwargs" not in payload


def test_unknown_component_provider_is_rejected() -> None:
    upstream = OpenAICompatibleUpstream(UpstreamConfig())

    with pytest.raises(UpstreamError, match="unknown upstream provider: missing"):
        upstream._build_request(
            messages=[],
            component=ComponentConfig(provider="missing"),
        )
