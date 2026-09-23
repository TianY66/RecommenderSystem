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
        if isinstance(self.response, list):
            return self.response.pop(0)
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


def test_stateless_responses_provider_resends_full_tool_call_history():
    first = SimpleNamespace(
        id="deepseek-r1",
        status="completed",
        output_text="",
        output=[
            SimpleNamespace(
                type="reasoning",
                content=[SimpleNamespace(type="reasoning_text", text="find candidates")],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-search",
                name="search_catalog",
                arguments='{"query":"算法"}',
            ),
        ],
        usage=None,
    )
    second = SimpleNamespace(
        id="deepseek-r2",
        status="completed",
        output_text="已检索。",
        output=[],
        usage=None,
    )
    client, endpoint = fake_client([first, second])
    provider = OpenAIResponsesProvider(client, model_name="deepseek-flash", stateless=True)
    user_input = [{"role": "user", "content": "搜索算法书"}]
    tools = [{"type": "function", "name": "search_catalog", "strict": True}]

    first_turn = provider.create_response(
        input_items=user_input,
        tools=tools,
        previous_response_id=None,
        instructions="instructions",
    )
    provider.create_response(
        input_items=[
            {
                "type": "function_call_output",
                "call_id": first_turn.tool_calls[0].call_id,
                "output": '{"candidates":[]}',
            }
        ],
        tools=tools,
        previous_response_id=first_turn.response_id,
        instructions="instructions",
    )

    replayed_input = endpoint.requests[1]["input"]
    assert [item.get("type", "message") for item in replayed_input] == [
        "message",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert replayed_input[1]["content"][0]["text"] == "find candidates"
    assert "previous_response_id" not in endpoint.requests[1]
    assert "parallel_tool_calls" not in endpoint.requests[0]
    assert endpoint.requests[0]["tools"] == [
        {"type": "function", "name": "search_catalog"}
    ]


def test_stateless_provider_rejects_tool_calls_without_response_id():
    client, _ = fake_client(
        SimpleNamespace(
            id=None,
            status="completed",
            output_text="",
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call-search",
                    name="search_catalog",
                    arguments='{"query":"算法"}',
                )
            ],
            usage=None,
        )
    )
    provider = OpenAIResponsesProvider(client, model_name="deepseek-flash", stateless=True)

    with pytest.raises(ValueError, match="without a response ID"):
        provider.create_response(
            input_items=[{"role": "user", "content": "搜索算法书"}],
            tools=[{"type": "function", "name": "search_catalog"}],
            previous_response_id=None,
            instructions="Use tools.",
        )


def test_deepseek_provider_reads_its_own_environment_names_and_stateless_mode():
    client, _ = fake_client()
    client_options = {}

    def client_factory(**kwargs):
        client_options.update(kwargs)
        return client

    provider = OpenAIResponsesProvider.from_env(
        {
            "DEEPSEEK_API_KEY": "not-a-real-secret",
            "DEEPSEEK_MODEL": "deepseek-flash",
        },
        provider="deepseek",
        client_factory=client_factory,
    )

    assert provider.model_name == "deepseek-flash"
    assert provider.stateless is True
    assert client_options == {
        "api_key": "not-a-real-secret",
        "base_url": "https://api.deepseek.com",
    }
