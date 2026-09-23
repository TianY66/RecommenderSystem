from types import SimpleNamespace

import pytest

from book_agent.providers import OpenAIResponsesProvider


class FakeResponsesEndpoint:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def fake_client(response=None, error=None):
    endpoint = FakeResponsesEndpoint(response=response, error=error)
    return SimpleNamespace(responses=endpoint), endpoint


def test_responses_provider_normalizes_function_calls_usage_and_api_request():
    response = SimpleNamespace(
        id="resp-1",
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="search_catalog",
                arguments='{"query":"算法"}',
            )
        ],
        usage=SimpleNamespace(input_tokens=22, output_tokens=5, total_tokens=27),
    )
    client, endpoint = fake_client(response)
    provider = OpenAIResponsesProvider(client, model_name="test-model")

    turn = provider.create_response(
        input_items=[{"role": "user", "content": "找算法书"}],
        tools=[{"type": "function", "name": "search_catalog"}],
        previous_response_id=None,
        instructions="Use tools.",
    )

    assert turn.response_id == "resp-1"
    assert turn.tool_calls[0].call_id == "call-1"
    assert turn.tool_calls[0].arguments == '{"query":"算法"}'
    assert turn.usage == {"input_tokens": 22, "output_tokens": 5, "total_tokens": 27}
    request = endpoint.requests[0]
    assert request["model"] == "test-model"
    assert request["parallel_tool_calls"] is False
    assert request["previous_response_id"] is None


def test_responses_provider_normalizes_final_answer_and_refusal():
    client, _ = fake_client(
        SimpleNamespace(
            id="resp-final",
            status="completed",
            output_text="可以为你搜索图书。",
            output=[],
            usage=None,
        )
    )
    provider = OpenAIResponsesProvider(client, model_name="test-model")
    turn = provider.create_response(
        input_items=[], tools=[], previous_response_id="resp-prev", instructions=""
    )
    assert turn.output_text == "可以为你搜索图书。"

    refusal_client, _ = fake_client(
        SimpleNamespace(
            id="resp-refusal",
            status="completed",
            output_text="",
            output=[SimpleNamespace(type="refusal", refusal="无法协助完成此请求。")],
            usage=None,
        )
    )
    refusal = OpenAIResponsesProvider(refusal_client, model_name="test-model").create_response(
        input_items=[], tools=[], previous_response_id=None, instructions=""
    )
    assert refusal.output_text == "无法协助完成此请求。"


def test_responses_provider_requires_key_and_model_before_client_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIResponsesProvider.from_env()


def test_responses_provider_preserves_api_errors_for_agent_error_handling():
    client, _ = fake_client(error=RuntimeError("service unavailable"))
    provider = OpenAIResponsesProvider(client, model_name="test-model")

    with pytest.raises(RuntimeError, match="service unavailable"):
        provider.create_response(
            input_items=[], tools=[], previous_response_id=None, instructions=""
        )
