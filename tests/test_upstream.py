from __future__ import annotations

from qwen_agent_proxy.config import UpstreamConfig, default_settings
from qwen_agent_proxy.upstream import OpenAICompatibleUpstream


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
