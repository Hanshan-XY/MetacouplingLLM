"""OpenAI GPT-6 models and an adapter-wide reasoning effort.

OpenAI's "Using GPT-6" guide: gpt-6-astra and gpt-6.1-sol take no ``"none"``
effort; while a model reasons the API rejects ``temperature``, and an
output-token cap counts reasoning tokens too, so a cap sized for the visible
answer can end a response during reasoning.  Chat Completions stops at
``"xhigh"`` for gpt-6.1-sol, so an adapter with an effort uses the Responses
API, which takes ``"max"``.
"""

from __future__ import annotations

from types import SimpleNamespace

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge.websearch import OpenAIWebSearchBackend
from metacouplingllm.llm.client import Message, OpenAIAdapter

ANSWER = "### 1. Coupling Classification\nTelecoupling."
SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "signals", "strict": True, "schema": {"type": "object"}},
}


class _FakeOpenAI:
    """Records Chat Completions and Responses requests; can fail the first Responses call."""

    def __init__(self, fail_first_response: Exception | None = None) -> None:
        self.chat_calls: list[dict] = []
        self.response_calls: list[dict] = []
        chat_calls, response_calls = self.chat_calls, self.response_calls
        pending = [fail_first_response] if fail_first_response else []

        class _Completions:
            @staticmethod
            def create(**kwargs):
                chat_calls.append(kwargs)
                message = SimpleNamespace(content=ANSWER)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

        class _Responses:
            @staticmethod
            def create(**kwargs):
                response_calls.append(kwargs)
                if pending:
                    raise pending.pop()
                usage = SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18)
                return SimpleNamespace(output_text=ANSWER, usage=usage, status="completed")

        self.chat = SimpleNamespace(completions=_Completions())
        self.responses = _Responses()


def _chat(model: str, effort: str | None = None, client: _FakeOpenAI | None = None, **chat_kwargs):
    client = client or _FakeOpenAI()
    reply = OpenAIAdapter(client, model=model, reasoning_effort=effort).chat(
        [Message(role="system", content="Be brief."), Message(role="user", content="ping")],
        **chat_kwargs,
    )
    return client, reply


class TestAdapter:
    def test_an_effort_goes_through_the_responses_api(self):
        client, reply = _chat("gpt-6.1-sol", "max", temperature=0.0, max_tokens=8192,
                              response_format=SCHEMA_FORMAT)
        assert client.chat_calls == []
        (sent,) = client.response_calls
        assert sent["model"] == "gpt-6.1-sol"
        assert sent["reasoning"] == {"effort": "max"}
        assert sent["input"] == [{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "ping"}]
        assert "temperature" not in sent and "max_output_tokens" not in sent
        assert sent["text"] == {"format": {"type": "json_schema", "name": "signals", "strict": True,
                                           "schema": {"type": "object"}}}
        assert reply.content == ANSWER
        assert reply.usage == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

    def test_effort_none_keeps_sampling_and_the_cap(self):
        client, _ = _chat("gpt-6.1-terra", "none", temperature=0.2, max_tokens=100)
        (sent,) = client.response_calls
        assert sent["temperature"] == 0.2 and sent["max_output_tokens"] == 100

    def test_gpt6_without_an_effort_uses_chat_completions_without_temperature_or_cap(self):
        client, _ = _chat("gpt-6.1-sol", temperature=0.7, max_tokens=8192)
        assert client.response_calls == []
        (sent,) = client.chat_calls
        assert "temperature" not in sent
        assert "max_tokens" not in sent and "max_completion_tokens" not in sent
        assert "reasoning_effort" not in sent

    def test_other_models_are_unchanged(self):
        client, _ = _chat("gpt-4o-mini", temperature=0.3, max_tokens=50)
        (sent,) = client.chat_calls
        assert sent["temperature"] == 0.3 and sent["max_tokens"] == 50
        client, _ = _chat("gpt-5.5", temperature=0.3, max_tokens=50)
        (sent,) = client.chat_calls
        assert sent["temperature"] == 1 and sent["max_completion_tokens"] == 50

    def test_a_rate_limit_error_is_retried(self, monkeypatch):
        monkeypatch.setattr("metacouplingllm.llm.client.time.sleep", lambda s: None)
        client = _FakeOpenAI(fail_first_response=Exception("Error code: 429 - rate limit reached"))
        _, reply = _chat("gpt-6.1-sol", "max", client=client)
        assert len(client.response_calls) == 2 and reply.content == ANSWER

    def test_effort_is_exposed(self):
        assert OpenAIAdapter(_FakeOpenAI(), model="gpt-6.1-sol", reasoning_effort="max").reasoning_effort == "max"
        assert OpenAIAdapter(_FakeOpenAI()).reasoning_effort is None


class _FakeResponses:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(output_text='{"results": []}', model_dump=lambda: {"output": []})


class TestWebSearchBackend:
    def test_an_explicit_effort_leaves_the_output_ceiling_to_the_model(self):
        client = SimpleNamespace(responses=_FakeResponses())
        OpenAIWebSearchBackend(client=client, model="gpt-6.1-sol", reasoning="max").search(
            "avocado production in Mexico", max_results=25,
        )
        assert client.responses.kwargs["reasoning"] == {"effort": "max"}
        assert "max_output_tokens" not in client.responses.kwargs

    def test_the_default_keeps_the_ceiling(self):
        client = SimpleNamespace(responses=_FakeResponses())
        OpenAIWebSearchBackend(client=client, model="gpt-5.5").search("avocado production in Mexico", max_results=25)
        assert "reasoning" not in client.responses.kwargs
        assert client.responses.kwargs["max_output_tokens"] >= 25000


def test_the_assistant_gives_its_web_search_the_adapter_effort(monkeypatch, tmp_path):
    seen = []

    def fake_search_web(query, max_results=5, backend=None, metadata=None, **kwargs):
        seen.append(backend)
        return []

    monkeypatch.setattr("metacouplingllm.knowledge.websearch.search_web", fake_search_web)
    client = _FakeOpenAI()
    adapter = OpenAIAdapter(client, model="gpt-6.1-sol", reasoning_effort="max")
    advisor = MetacouplingAssistant(
        adapter, web_search=True, web_structured_extraction=False,
        generate_abstract=False, maritime_adjacency=True, trace=True, trace_dir=tmp_path,
    )
    advisor.analyze("Impact of avocado production and trade in Mexico on sustainability")
    (backend,) = seen
    assert isinstance(backend, OpenAIWebSearchBackend)
    assert backend.model == "gpt-6.1-sol" and backend.reasoning == "max"
    assert client.chat_calls == [] and client.response_calls  # the analysis went through Responses

    # The run's configuration records the effort and the adjacency settings.
    config = next(tmp_path.rglob("00_run_config.md")).read_text(encoding="utf-8")
    for row in ("| `reasoning_effort` | max |", "| `maritime_adjacency` | True |",
                "| `coupling_standard` | lenient |", "| `de_facto_borders` | True |"):
        assert row in config, row
