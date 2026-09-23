"""Provider-neutral response objects used by the book agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from collections.abc import Mapping
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ProviderTurn:
    response_id: str | None
    output_text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)
    status: str | None = None


class LLMProvider(Protocol):
    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str,
    ) -> ProviderTurn:
        """Create one model turn and normalize its output."""


class OpenAIResponsesProvider:
    """Thin adapter for the OpenAI Responses API.

    The client is injectable so request/response handling can be tested without
    network access. ``from_env`` creates a real SDK client only when configured.
    """

    def __init__(self, client: Any, *, model_name: str) -> None:
        self.client = client
        self.model_name = model_name

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "OpenAIResponsesProvider":
        settings = os.environ if environ is None else environ
        api_key = settings.get("OPENAI_API_KEY", "").strip()
        model_name = settings.get("OPENAI_MODEL", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for real-model evaluation")
        if not model_name:
            raise ValueError("OPENAI_MODEL is required for real-model evaluation")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI Python SDK is missing; install requirements-agent.txt"
            ) from exc

        base_url = settings.get("OPENAI_BASE_URL", "").strip()
        client_options: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        return cls(OpenAI(**client_options), model_name=model_name)

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str,
    ) -> ProviderTurn:
        response = self.client.responses.create(
            model=self.model_name,
            input=input_items,
            tools=tools,
            instructions=instructions,
            previous_response_id=previous_response_id,
            parallel_tool_calls=False,
        )
        normalized_calls = tuple(
            ToolCall(
                call_id=str(_read(item, "call_id", "")),
                name=str(_read(item, "name", "")),
                arguments=str(_read(item, "arguments", "")),
            )
            for item in (_read(response, "output", []) or [])
            if _read(item, "type") == "function_call"
        )
        text = _read(response, "output_text")
        if not text:
            refusals = [
                str(_read(item, "refusal", ""))
                for item in (_read(response, "output", []) or [])
                if _read(item, "type") == "refusal" and _read(item, "refusal")
            ]
            text = "\n".join(refusals) or None
        raw_usage = _read(response, "usage")
        usage = {
            key: value
            for key in ("input_tokens", "output_tokens", "total_tokens")
            if (value := _read(raw_usage, key)) is not None
        }
        return ProviderTurn(
            response_id=_read(response, "id"),
            output_text=text,
            tool_calls=normalized_calls,
            usage=usage,
            status=_read(response, "status"),
        )


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
