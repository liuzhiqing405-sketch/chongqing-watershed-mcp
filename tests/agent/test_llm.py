"""The optional language-model client: configuration, transport and failure.

Nothing here needs a real endpoint. The transport is stubbed so the client's
own behaviour — request shape, key handling, error mapping — is what is tested.
"""

from __future__ import annotations

import json

import pytest

from slope_evaluation_agent.config import LLMSettings
from slope_evaluation_agent.llm import (
    LLMRequestFailed,
    LLMUnavailable,
    OpenAICompatibleClient,
    build_llm_client,
    extract_json,
)


class FakeResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeHTTPX:
    """Stands in for the ``httpx`` module for the duration of a test."""

    def __init__(self, response=None, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.requests: list[dict] = []

    def AsyncClient(self, **kwargs):  # noqa: N802 - mirrors the httpx API
        outer = self

        class _Client:
            def __init__(self, **client_kwargs) -> None:
                self.kwargs = client_kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                return None

            async def post(self, url, headers=None, json=None):
                outer.requests.append({"url": url, "headers": headers or {}, "json": json})
                if outer.raises is not None:
                    raise outer.raises
                return outer.response

        return _Client(**kwargs)


def reply(content: str, status_code: int = 200) -> FakeResponse:
    return FakeResponse({"choices": [{"message": {"content": content}}]}, status_code)


def settings(**overrides) -> LLMSettings:
    payload = {"base_url": "https://example.invalid/v1", "model": "test-model",
               "api_key": "sk-test-key-not-real"}
    payload.update(overrides)
    return LLMSettings.model_validate(payload)


def install(monkeypatch, fake: FakeHTTPX) -> None:
    import sys
    import types

    module = types.ModuleType("httpx")
    module.AsyncClient = fake.AsyncClient
    monkeypatch.setitem(sys.modules, "httpx", module)


class TestAvailability:
    def test_no_client_without_a_base_url_and_model(self):
        assert build_llm_client(LLMSettings()) is None

    def test_no_client_when_explicitly_disabled(self):
        assert build_llm_client(settings(enabled=False)) is None

    def test_a_client_when_configured(self):
        client = build_llm_client(settings())
        assert isinstance(client, OpenAICompatibleClient)

    @pytest.mark.asyncio
    async def test_calling_without_configuration_is_a_clear_error(self):
        client = OpenAICompatibleClient(LLMSettings())
        with pytest.raises(LLMUnavailable):
            await client.complete_json(system="s", user="u")


class TestRequestShape:
    @pytest.mark.parametrize("provider,thinking", [
        ("deepseek", "disabled"), ("deepseek", "enabled"), ("openai-compatible", "disabled"),
    ])
    @pytest.mark.asyncio
    async def test_thinking_toggle_is_sent_only_to_deepseek(self, monkeypatch, provider, thinking):
        fake = FakeHTTPX(reply('{"answer":"地下水会改变有效应力。"}'))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings(provider=provider, deepseek_thinking=thinking))
        await client.complete_json(system="JSON", user="水流如何影响坡面稳定性")
        body = fake.requests[0]["json"]
        assert body.get("thinking") == ({"type": thinking} if provider == "deepseek" else None)

    @pytest.mark.asyncio
    async def test_the_request_is_openai_shaped(self, monkeypatch):
        fake = FakeHTTPX(reply('{"intent": "unknown"}'))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings())
        await client.complete_json(system="SYSTEM", user="USER")
        request = fake.requests[0]
        assert request["url"] == "https://example.invalid/v1/chat/completions"
        assert request["json"]["model"] == "test-model"
        assert request["json"]["temperature"] == 0.0
        assert request["json"]["response_format"] == {"type": "json_object"}
        assert request["json"]["messages"][0] == {"role": "system", "content": "SYSTEM"}
        assert request["json"]["messages"][1] == {"role": "user", "content": "USER"}

    @pytest.mark.asyncio
    async def test_the_key_travels_in_the_authorization_header_only(self, monkeypatch):
        fake = FakeHTTPX(reply('{"intent": "unknown"}'))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings())
        await client.complete_json(system="s", user="u")
        request = fake.requests[0]
        assert request["headers"]["Authorization"] == "Bearer sk-test-key-not-real"
        assert "sk-test-key-not-real" not in json.dumps(request["json"])

    @pytest.mark.asyncio
    async def test_a_trailing_slash_in_the_base_url_is_tolerated(self, monkeypatch):
        fake = FakeHTTPX(reply('{"intent": "unknown"}'))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings(base_url="https://example.invalid/v1/"))
        await client.complete_json(system="s", user="u")
        assert fake.requests[0]["url"] == "https://example.invalid/v1/chat/completions"


class TestFailures:
    @pytest.mark.asyncio
    async def test_truncated_thinking_only_response_is_retried_with_more_budget(self, monkeypatch):
        first = FakeResponse({"choices": [{"finish_reason": "length", "message": {
            "content": None, "reasoning_content": "internal reasoning is not an answer"}}]})
        fake = FakeHTTPX(first)
        responses = iter([first, reply('{"answer":"水流改变孔压，并可侵蚀坡脚。"}')])
        monkeypatch.setattr(first, "json", lambda: next(responses)._payload)
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings(provider="deepseek", maximum_tokens=1200, max_retries=1))
        result = await client.complete_json(system="JSON", user="水流如何影响坡面稳定性")
        assert result["answer"] == "水流改变孔压，并可侵蚀坡脚。"
        assert len(fake.requests) == 2
        assert fake.requests[-1]["json"]["max_tokens"] == 2400

    @pytest.mark.parametrize("content,finish,reason", [
        (None, "length", "response_truncated"), (None, "stop", "empty_response"),
        ("", "stop", "empty_response"), ("{", "stop", "invalid_json"),
    ])
    @pytest.mark.asyncio
    async def test_failure_reason_is_specific_and_does_not_return_reasoning(self, monkeypatch, content, finish, reason):
        fake = FakeHTTPX(FakeResponse({"choices": [{"finish_reason": finish, "message": {
            "content": content, "reasoning_content": "private chain, not final answer"}}]}))
        install(monkeypatch, fake)
        with pytest.raises(LLMRequestFailed) as error:
            await OpenAICompatibleClient(settings(max_retries=0)).complete_json(system="JSON", user="u")
        assert error.value.reason == reason
        assert "private chain" not in str(error.value)

    @pytest.mark.asyncio
    async def test_an_http_error_is_reported_without_the_body(self, monkeypatch):
        fake = FakeHTTPX(reply("server said: sk-test-key-not-real is invalid", status_code=401))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings())
        with pytest.raises(LLMRequestFailed) as excinfo:
            await client.complete_json(system="s", user="u")
        assert "401" in str(excinfo.value)
        assert "sk-test-key-not-real" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_transport_error_is_reported(self, monkeypatch):
        fake = FakeHTTPX(raises=ConnectionError("no route to host"))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings(max_retries=0))
        with pytest.raises(LLMRequestFailed):
            await client.complete_json(system="s", user="u")

    @pytest.mark.asyncio
    async def test_a_malformed_response_is_reported(self, monkeypatch):
        fake = FakeHTTPX(FakeResponse({"nothing": "useful"}))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings())
        with pytest.raises(LLMRequestFailed):
            await client.complete_json(system="s", user="u")

    @pytest.mark.asyncio
    async def test_a_non_json_reply_is_reported(self, monkeypatch):
        fake = FakeHTTPX(reply("I cannot help with that."))
        install(monkeypatch, fake)
        client = OpenAICompatibleClient(settings())
        with pytest.raises(LLMRequestFailed):
            await client.complete_json(system="s", user="u")


class TestJSONExtraction:
    def test_fences_and_prose_are_tolerated(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
        assert extract_json('sure: {"a": 1}') == {"a": 1}
        assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_nested_braces_survive(self):
        assert extract_json('{"a": {"b": [1, 2]}}') == {"a": {"b": [1, 2]}}
