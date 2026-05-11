"""LLM provider abstraction.

The engine talks to a `Provider`. Two implementations ship:

* `AnthropicProvider`  — wraps `anthropic.AsyncAnthropic`. Preserves
  ephemeral prompt caching on the per-agent system prompt.
* `OpenAICompatProvider` — wraps `openai.AsyncOpenAI` pointed at any
  OpenAI-compatible endpoint. Covers Ollama, LM Studio, vLLM, llama.cpp,
  LocalAI, OpenAI itself, OpenRouter, Groq, Together, and custom URLs.
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Protocol


PROVIDER_PRESETS: dict[str, dict] = {
    "anthropic": {
        "kind": "anthropic",
        "label": "Anthropic Claude (cloud, paid)",
    },
    "ollama": {
        "kind": "openai_compat",
        "label": "Ollama (local, free)",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "lmstudio": {
        "kind": "openai_compat",
        "label": "LM Studio (local, free)",
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
    },
    "vllm": {
        "kind": "openai_compat",
        "label": "vLLM (local/self-hosted)",
        "base_url": "http://localhost:8000/v1",
        "api_key": "EMPTY",
    },
    "localai": {
        "kind": "openai_compat",
        "label": "LocalAI (local)",
        "base_url": "http://localhost:8080/v1",
        "api_key": "not-needed",
    },
    "openai": {
        "kind": "openai_compat",
        "label": "OpenAI (cloud, paid)",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "kind": "openai_compat",
        "label": "OpenRouter (cloud, pay-as-you-go)",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "groq": {
        "kind": "openai_compat",
        "label": "Groq (cloud, free tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "together": {
        "kind": "openai_compat",
        "label": "Together AI (cloud)",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "custom": {
        "kind": "openai_compat",
        "label": "Custom OpenAI-compatible endpoint",
    },
}


class Provider(Protocol):
    name: str

    def stream(
        self,
        *,
        system: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> AsyncIterator[str]: ...

    async def list_models(self) -> list[str]: ...

    async def close(self) -> None: ...


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic
        self._client = AsyncAnthropic(api_key=api_key)

    async def stream(
        self,
        *,
        system: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        async with self._client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk

    async def list_models(self) -> list[str]:
        return [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]

    async def close(self) -> None:
        await self._client.close()


class OpenAICompatProvider:
    def __init__(self, *, base_url: str, api_key: str, name: str = "openai_compat") -> None:
        from openai import AsyncOpenAI
        self.name = name
        self.base_url = base_url
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def stream(
        self,
        *,
        system: str,
        user_message: str,
        model: str,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def list_models(self) -> list[str]:
        result = await self._client.models.list()
        return sorted(m.id for m in result.data)

    async def close(self) -> None:
        await self._client.close()


def _resolve_api_key(preset: dict, override: str | None) -> str:
    if override:
        return override
    env_name = preset.get("api_key_env")
    if env_name:
        key = os.getenv(env_name)
        if not key:
            raise RuntimeError(
                f"Provider needs {env_name} in the environment (or pass --api-key)."
            )
        return key
    return preset.get("api_key", "not-needed")


def make_provider(
    provider: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Provider:
    """Build a Provider for `provider` (a key from PROVIDER_PRESETS)."""
    if provider not in PROVIDER_PRESETS:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Choose one of: {', '.join(PROVIDER_PRESETS)}"
        )
    preset = PROVIDER_PRESETS[provider]
    kind = preset["kind"]

    if kind == "anthropic":
        return AnthropicProvider(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    if kind == "openai_compat":
        resolved_base_url = base_url or preset.get("base_url")
        if not resolved_base_url:
            raise RuntimeError(
                f"Provider {provider!r} needs a base URL (pass --base-url or set it in agents.yaml)."
            )
        resolved_key = _resolve_api_key(preset, api_key)
        return OpenAICompatProvider(
            base_url=resolved_base_url,
            api_key=resolved_key,
            name=provider,
        )

    raise ValueError(f"Unknown provider kind: {kind}")
