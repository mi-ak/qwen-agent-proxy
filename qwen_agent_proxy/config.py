from __future__ import annotations

import copy
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import tomllib


@dataclass(slots=True)
class UpstreamConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "dummy"
    model: str = "Qwen/Qwen3-8B"
    thinking_param_style: str = "chat_template_kwargs"


@dataclass(slots=True)
class AgentConfig:
    public_model_id: str = "qwen-agent"
    max_tool_retries: int = 2
    parallel_tool_call: bool = False
    log_requests: bool = True
    log_upstream: bool = True


@dataclass(slots=True)
class ComponentConfig:
    provider: str = "default"
    model: str | None = None
    enable_thinking: bool = True
    temperature: float = 0.2
    max_tokens: int = 4096


@dataclass(slots=True)
class Settings:
    upstream: UpstreamConfig
    providers: dict[str, UpstreamConfig]
    agent: AgentConfig
    planner: ComponentConfig
    tool_caller: ComponentConfig
    finalizer: ComponentConfig


def default_settings() -> Settings:
    return Settings(
        upstream=UpstreamConfig(),
        providers={},
        agent=AgentConfig(),
        planner=ComponentConfig(enable_thinking=True, temperature=0.2, max_tokens=4096),
        tool_caller=ComponentConfig(enable_thinking=False, temperature=0.0, max_tokens=2048),
        finalizer=ComponentConfig(enable_thinking=True, temperature=0.3, max_tokens=8192),
    )


def _section_to_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid_names = {field.name for field in fields(instance)}
    for key, value in values.items():
        if key in valid_names:
            setattr(instance, key, value)
    return instance


def apply_config(settings: Settings, data: dict[str, Any]) -> Settings:
    section_names = (
        "upstream",
        "agent",
        "planner",
        "tool_caller",
        "finalizer",
    )
    for section in section_names:
        values = data.get(section)
        if isinstance(values, dict):
            _section_to_dataclass(getattr(settings, section), values)
    providers = data.get("providers")
    if isinstance(providers, dict):
        settings.providers = _providers_from_config(settings.upstream, providers)
    return settings


def _providers_from_config(
    default_provider: UpstreamConfig,
    values: dict[str, Any],
) -> dict[str, UpstreamConfig]:
    providers: dict[str, UpstreamConfig] = {}
    for name, provider_values in values.items():
        if not isinstance(name, str) or not isinstance(provider_values, dict):
            continue
        provider = copy.deepcopy(default_provider)
        providers[name] = _section_to_dataclass(provider, provider_values)
    return providers


def load_config(path: str | Path | None = None) -> Settings:
    settings = copy.deepcopy(default_settings())
    config_path = _resolve_config_path(path)
    if config_path is None:
        return settings

    with config_path.open("rb") as file:
        data = tomllib.load(file)
    return apply_config(settings, data)


def _resolve_config_path(path: str | Path | None) -> Path | None:
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.exists() else None

    env_path = os.environ.get("QWEN_AGENT_PROXY_CONFIG")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate

    local_path = Path.cwd() / "config.toml"
    if local_path.exists():
        return local_path

    return None
