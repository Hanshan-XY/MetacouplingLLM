"""Tests for first-class Gemini and Grok support.

Covers:
- GeminiAdapter: message conversion (system → system_instruction,
  assistant → model role), generation_config wiring, usage extraction,
  empty-response safety
- GrokAdapter: passes messages through to OpenAI-shaped chat API,
  uses configured Grok model, raw_client property
- GeminiWebSearchBackend: parses JSON from response.text, falls back
  to grounding_metadata.grounding_chunks, handles errors, caps results
- GrokWebSearchBackend: passes search_parameters via extra_body, parses
  JSON from message content, falls back to response.citations
- Auto-wiring: isinstance dispatch in MetacouplingAssistant routes to
  the right backend based on the adapter type
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from metacouplingllm import (
    GeminiAdapter,
    GeminiWebSearchBackend,
    GrokAdapter,
    GrokWebSearchBackend,
    LLMResponse,
    Message,
    MetacouplingAssistant,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class _RecordingGeminiClient:
    """Mimics the new ``google.genai.Client`` surface for tests.

    Records calls to ``client.models.generate_content`` and returns a
    configurable response.
    """

    def __init__(self, response_text: str = "Hi!", usage: dict | None = None,
                 grounding_chunks: list[Any] | None = None):
        self._text = response_text
        self._usage = usage or {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        }
        self._grounding_chunks = grounding_chunks or []
        self.calls: list[dict[str, Any]] = []

        self.models = SimpleNamespace(generate_content=self._generate_content)

    def _generate_content(self, **kwargs):
        self.calls.append(kwargs)
        usage_meta = SimpleNamespace(**self._usage)
        candidate = SimpleNamespace(
            grounding_metadata=SimpleNamespace(
                grounding_chunks=self._grounding_chunks
            )
        )
        return SimpleNamespace(
            text=self._text,
            usage_metadata=usage_meta,
            candidates=[candidate],
        )


class _RecordingOpenAIClient:
    """Mimics openai.OpenAI for both OpenAI and Grok adapter tests."""

    def __init__(self, content: str = "ok", citations: list[Any] | None = None):
        self._content = content
        self._citations = citations or []
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        msg = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=msg)
        usage = SimpleNamespace(
            prompt_tokens=8, completion_tokens=4, total_tokens=12
        )
        resp = SimpleNamespace(
            choices=[choice], usage=usage, citations=self._citations
        )
        return resp


# ---------------------------------------------------------------------------
# GeminiAdapter
# ---------------------------------------------------------------------------


class TestGeminiAdapter:
    def test_chat_extracts_system_messages_to_system_instruction(self):
        client = _RecordingGeminiClient(response_text="Hello back!")
        adapter = GeminiAdapter(client, model="gemini-2.5-flash")

        adapter.chat([
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi!"),
        ])

        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["model"] == "gemini-2.5-flash"
        # System instruction goes into the config, not the messages
        assert call["config"]["system_instruction"] == "You are helpful."
        # The conversation has only the user msg
        contents = call["contents"]
        assert len(contents) == 1
        assert contents[0]["role"] == "user"
        assert contents[0]["parts"][0]["text"] == "Hi!"

    def test_chat_converts_assistant_role_to_model(self):
        client = _RecordingGeminiClient(response_text="ok")
        adapter = GeminiAdapter(client)

        adapter.chat([
            Message(role="user", content="Q1"),
            Message(role="assistant", content="A1"),
            Message(role="user", content="Q2"),
        ])

        contents = client.calls[0]["contents"]
        assert [m["role"] for m in contents] == ["user", "model", "user"]

    def test_chat_calls_generate_content_with_temperature_and_max_tokens(self):
        client = _RecordingGeminiClient(response_text="ok")
        adapter = GeminiAdapter(client)

        adapter.chat(
            [Message(role="user", content="Hi")],
            temperature=0.3,
            max_tokens=512,
        )

        cfg = client.calls[0]["config"]
        assert cfg["temperature"] == 0.3
        assert cfg["max_output_tokens"] == 512

    def test_chat_returns_llmresponse_with_token_usage(self):
        client = _RecordingGeminiClient(
            response_text="answer text",
            usage={
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "total_token_count": 150,
            },
        )
        adapter = GeminiAdapter(client)

        result = adapter.chat([Message(role="user", content="Hi")])

        assert isinstance(result, LLMResponse)
        assert result.content == "answer text"
        assert result.usage == {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_chat_handles_empty_response_safely(self):
        # Some Gemini responses raise on .text (e.g., when blocked by
        # safety filters). The adapter must not crash.
        class _BlockingResponse:
            @property
            def text(self):
                raise ValueError("Response blocked")

            usage_metadata = SimpleNamespace(
                prompt_token_count=5,
                candidates_token_count=0,
                total_token_count=5,
            )
            candidates = []

        class _BlockingClient:
            models = SimpleNamespace(
                generate_content=lambda **kw: _BlockingResponse()
            )

        adapter = GeminiAdapter(_BlockingClient())
        result = adapter.chat([Message(role="user", content="x")])
        assert result.content == ""
        assert result.usage["input_tokens"] == 5


# ---------------------------------------------------------------------------
# GrokAdapter
# ---------------------------------------------------------------------------


class TestGrokAdapter:
    def test_chat_passes_through_messages_unchanged(self):
        client = _RecordingOpenAIClient(content="grok says hi")
        adapter = GrokAdapter(client, model="grok-3")

        adapter.chat([
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
        ])

        msgs = client.calls[0]["messages"]
        assert msgs == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_chat_calls_chat_completions_create_with_grok_model(self):
        client = _RecordingOpenAIClient(content="ok")
        adapter = GrokAdapter(client, model="grok-3")
        adapter.chat([Message(role="user", content="x")], temperature=0.5)

        assert client.calls[0]["model"] == "grok-3"
        assert client.calls[0]["temperature"] == 0.5

    def test_raw_client_property_exposes_underlying_openai_client(self):
        client = _RecordingOpenAIClient()
        adapter = GrokAdapter(client, model="grok-3")
        assert adapter.raw_client is client
        assert adapter.model == "grok-3"


# ---------------------------------------------------------------------------
# GeminiWebSearchBackend
# ---------------------------------------------------------------------------


class TestGeminiWebSearchBackend:
    def test_search_returns_results_from_json_response(self):
        json_text = '''
        {"results":[
            {"title":"Paper A","url":"https://a.example","snippet":"snip A"},
            {"title":"Paper B","url":"https://b.example","snippet":"snip B"}
        ]}
        '''
        client = _RecordingGeminiClient(response_text=json_text)
        backend = GeminiWebSearchBackend(client=client)

        results = backend.search("metacoupling", max_results=5)
        assert len(results) == 2
        assert results[0]["title"] == "Paper A"
        assert results[0]["url"] == "https://a.example"
        assert results[1]["snippet"] == "snip B"

    def test_search_falls_back_to_grounding_chunks_when_json_invalid(self):
        # Model returns non-JSON text, but grounding_chunks have URLs
        web1 = SimpleNamespace(uri="https://x.example", title="X")
        web2 = SimpleNamespace(uri="https://y.example", title="Y")
        chunk1 = SimpleNamespace(web=web1)
        chunk2 = SimpleNamespace(web=web2)
        client = _RecordingGeminiClient(
            response_text="not json at all",
            grounding_chunks=[chunk1, chunk2],
        )
        backend = GeminiWebSearchBackend(client=client)

        results = backend.search("query", max_results=5)
        assert len(results) == 2
        assert results[0]["url"] == "https://x.example"
        assert results[0]["title"] == "X"

    def test_search_returns_empty_list_on_exception(self):
        class _BrokenClient:
            models = SimpleNamespace(
                generate_content=lambda **kw: (_ for _ in ()).throw(
                    RuntimeError("Gemini down")
                )
            )

        backend = GeminiWebSearchBackend(client=_BrokenClient())
        assert backend.search("anything", max_results=5) == []

    def test_search_caps_results_at_max_results(self):
        json_text = '''
        {"results":[
            {"title":"1","url":"https://1.x","snippet":""},
            {"title":"2","url":"https://2.x","snippet":""},
            {"title":"3","url":"https://3.x","snippet":""},
            {"title":"4","url":"https://4.x","snippet":""},
            {"title":"5","url":"https://5.x","snippet":""}
        ]}
        '''
        client = _RecordingGeminiClient(response_text=json_text)
        backend = GeminiWebSearchBackend(client=client)
        results = backend.search("query", max_results=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# GrokWebSearchBackend
# ---------------------------------------------------------------------------


class TestGrokWebSearchBackend:
    def test_search_passes_search_parameters_to_xai_api(self):
        client = _RecordingOpenAIClient(content='{"results":[]}')
        backend = GrokWebSearchBackend(client=client, model="grok-3")
        backend.search("metacoupling", max_results=5)

        assert "extra_body" in client.calls[0]
        params = client.calls[0]["extra_body"]["search_parameters"]
        assert params["mode"] == "auto"
        assert params["max_search_results"] == 15
        assert {"type": "web"} in params["sources"]
        assert {"type": "x"} in params["sources"]

    def test_search_returns_results_from_json_response(self):
        json_text = '''
        {"results":[
            {"title":"R1","url":"https://r1.x","snippet":"s1"}
        ]}
        '''
        client = _RecordingOpenAIClient(content=json_text)
        backend = GrokWebSearchBackend(client=client)
        results = backend.search("query", max_results=5)
        assert len(results) == 1
        assert results[0]["url"] == "https://r1.x"

    def test_search_falls_back_to_citations_when_json_invalid(self):
        # Plain text response + URL strings as citations
        client = _RecordingOpenAIClient(
            content="freeform text, no JSON here",
            citations=["https://cite1.x", "https://cite2.x"],
        )
        backend = GrokWebSearchBackend(client=client)
        results = backend.search("query", max_results=5)
        assert len(results) == 2
        urls = [r["url"] for r in results]
        assert urls == ["https://cite1.x", "https://cite2.x"]

    def test_search_returns_empty_list_on_exception(self):
        class _BrokenClient:
            chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: (_ for _ in ()).throw(
                        RuntimeError("xAI down")
                    )
                )
            )

        backend = GrokWebSearchBackend(client=_BrokenClient())
        assert backend.search("anything", max_results=5) == []


# ---------------------------------------------------------------------------
# Auto-wiring in MetacouplingAssistant
# ---------------------------------------------------------------------------


class TestAutoWiring:
    def test_gemini_adapter_triggers_gemini_web_backend(
        self, mock_rag_engine, monkeypatch
    ):
        """When the advisor's client is a GeminiAdapter, web search must
        construct a GeminiWebSearchBackend (not a DDG fallback)."""
        from metacouplingllm.knowledge import websearch as ws

        captured: dict[str, Any] = {}

        def _fake_search_web(query, max_results=5, backend=None, metadata=None):
            captured["backend"] = backend
            return []

        monkeypatch.setattr(ws, "search_web", _fake_search_web)

        gem_client = _RecordingGeminiClient(response_text="answer [1]")
        adapter = GeminiAdapter(gem_client, model="gemini-2.5-flash")

        advisor = MetacouplingAssistant(
            llm_client=adapter,
            max_examples=0,
            coupling_analysis=False,
            web_search=True,
        )
        advisor._rag_engine = mock_rag_engine
        advisor.analyze("anything")

        assert isinstance(captured["backend"], GeminiWebSearchBackend)
        assert captured["backend"].model == "gemini-2.5-flash"

    def test_grok_adapter_triggers_grok_web_backend(
        self, mock_rag_engine, monkeypatch
    ):
        from metacouplingllm.knowledge import websearch as ws

        captured: dict[str, Any] = {}

        def _fake_search_web(query, max_results=5, backend=None, metadata=None):
            captured["backend"] = backend
            return []

        monkeypatch.setattr(ws, "search_web", _fake_search_web)

        client = _RecordingOpenAIClient(content="answer [1]")
        adapter = GrokAdapter(client, model="grok-3")

        advisor = MetacouplingAssistant(
            llm_client=adapter,
            max_examples=0,
            coupling_analysis=False,
            web_search=True,
        )
        advisor._rag_engine = mock_rag_engine
        advisor.analyze("anything")

        assert isinstance(captured["backend"], GrokWebSearchBackend)
        assert captured["backend"].model == "grok-3"
