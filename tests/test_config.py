from __future__ import annotations

from qwen_agent_proxy.config import apply_config, default_settings


def test_named_providers_inherit_upstream_defaults() -> None:
    settings = apply_config(
        default_settings(),
        {
            "upstream": {
                "base_url": "http://default.example/v1",
                "api_key": "shared-key",
                "model": "Qwen/Qwen3-8B",
                "thinking_param_style": "disabled",
            },
            "providers": {
                "tools": {
                    "base_url": "http://tools.example/v1",
                    "model": "Qwen/Qwen3-4B",
                },
            },
            "tool_caller": {
                "provider": "tools",
                "model": "Qwen/Qwen3-4B-Instruct",
            },
        },
    )

    assert settings.providers["tools"].base_url == "http://tools.example/v1"
    assert settings.providers["tools"].api_key == "shared-key"
    assert settings.providers["tools"].thinking_param_style == "disabled"
    assert settings.tool_caller.provider == "tools"
    assert settings.tool_caller.model == "Qwen/Qwen3-4B-Instruct"


def test_parallel_tool_call_defaults_to_disabled() -> None:
    assert default_settings().agent.parallel_tool_call is False
