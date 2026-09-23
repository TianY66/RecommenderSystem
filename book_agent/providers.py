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

    def __init__(self, client: Any, *, model_name: str, stateless: bool = False) -> None:
        self.client = client
        self.model_name = model_name
        self.stateless = stateless
        self._histories: dict[str, list[dict[str, Any]]] = {}

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        provider: str = "openai",
        client_factory: Any = None,
    ) -> "OpenAIResponsesProvider":
        settings = os.environ if environ is None else environ
        provider = provider.casefold()
        if provider == "deepseek":
            api_key = settings.get("DEEPSEEK_API_KEY", "").strip()
            model_name = settings.get("DEEPSEEK_MODEL", "deepseek-flash").strip()
            base_url = (
                settings.get("DEEPSEEK_BASE_URL", "").strip()
                or "https://api.deepseek.com"
            )
            key_name = "DEEPSEEK_API_KEY"
            stateless = True
        elif provider == "openai":
            api_key = settings.get("OPENAI_API_KEY", "").strip()
            model_name = settings.get("OPENAI_MODEL", "").strip()
            base_url = settings.get("OPENAI_BASE_URL", "").strip()
            key_name = "OPENAI_API_KEY"
            stateless = False
        else:
            raise ValueError("provider must be 'openai' or 'deepseek'")
        if not api_key:
            raise ValueError(f"{key_name} is required for real-model evaluation")
        if not model_name:
            raise ValueError(f"{provider.upper()}_MODEL is required for real-model evaluation")

        if client_factory is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI Python SDK is missing; install requirements-agent.txt"
                ) from exc
            client_factory = OpenAI

        client_options: dict[str, str] = {"api_key": api_key}
        if base_url:
            client_options["base_url"] = base_url
        return cls(
            client_factory(**client_options),
            model_name=model_name,
            stateless=stateless,
        )

    def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        previous_response_id: str | None,
        instructions: str,
    ) -> ProviderTurn:
        request_input = input_items
        if self.stateless:
            if previous_response_id is None:
                request_input = list(input_items)
            else:
                history = self._histories.pop(previous_response_id, None)
                if history is None:
                    raise ValueError("Missing local conversation history for stateless provider")
                request_input = history + list(input_items)
            request = {
                "model": self.model_name,
                "input": request_input,
                "tools": _stateless_tools(tools),
                "instructions": instructions,
            }
        else:
            request = {
                "model": self.model_name,
                "input": input_items,
                "tools": tools,
                "instructions": instructions,
                "previous_response_id": previous_response_id,
                "parallel_tool_calls": False,
            }
        response = self.client.responses.create(**request)
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
        response_id = _read(response, "id")
        if self.stateless and normalized_calls and not response_id:
            raise ValueError("Stateless provider returned tool calls without a response ID")
        if self.stateless and normalized_calls and response_id:
            self._histories[str(response_id)] = request_input + _response_input_items(
                _read(response, "output", []) or []
            )
        return ProviderTurn(
            response_id=response_id,
            output_text=text,
            tool_calls=normalized_calls,
            usage=usage,
            status=_read(response, "status"),
        )


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _response_input_items(output_items: list[Any]) -> list[dict[str, Any]]:
    """Convert assistant output items to the supported stateless input history form."""

    input_items: list[dict[str, Any]] = []
    for item in output_items:
        item_type = _read(item, "type")
        if item_type == "function_call":
            input_items.append(
                {
                    "type": "function_call",
                    "call_id": _read(item, "call_id"),
                    "name": _read(item, "name"),
                    "arguments": _read(item, "arguments", ""),
                }
            )
        elif item_type == "reasoning":
            content = [
                {
                    "type": "reasoning_text",
                    "text": _read(part, "text", ""),
                }
                for part in (_read(item, "content", []) or [])
                if _read(part, "type") == "reasoning_text"
            ]
            if content:
                input_items.append({"type": "reasoning", "content": content})
        elif item_type == "message":
            content = []
            for part in (_read(item, "content", []) or []):
                part_type = _read(part, "type")
                if part_type == "output_text":
                    content.append(
                        {"type": "output_text", "text": _read(part, "text", "")}
                    )
                elif part_type == "refusal" and _read(part, "refusal"):
                    content.append(
                        {"type": "output_text", "text": _read(part, "refusal")}
                    )
            if content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                )
    return input_items


def _stateless_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Send only the Responses API fields DeepSeek documents for function tools."""

    return [
        {key: value for key, value in tool.items() if key != "strict"}
        for tool in tools
    ]
