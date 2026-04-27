"""
Shared pytest fixtures for the metacoupling test suite.

Most existing tests instantiate their own mocks inline; this conftest is
new and exists primarily to support `tests/test_rag_pipeline.py`. The
fixtures here are scoped narrowly so they do not affect tests that don't
opt in by name.
"""

from __future__ import annotations

from typing import Any

import pytest

from metacouplingllm.core import MetacouplingAssistant
from metacouplingllm.knowledge.rag import RetrievalResult, TextChunk
from metacouplingllm.llm.client import LLMResponse, Message


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


class _RecordingMockLLMClient:
    """Mock LLM client that returns a configurable response and records calls.

    Distinct from the inline ``MockLLMClient`` in ``tests/test_core.py`` so
    the new RAG-pipeline tests have full control over what the LLM "says"
    on each turn without coupling to the legacy mock's coffee-trade fixture.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default_response: str = "",
    ) -> None:
        self._responses = list(responses or [])
        self._default = default_response
        self.call_count = 0
        self.calls: list[list[Message]] = []
        self.last_messages: list[Message] | None = None

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        self.last_messages = messages
        self.calls.append(list(messages))
        idx = self.call_count
        self.call_count += 1
        if idx < len(self._responses):
            content = self._responses[idx]
        else:
            content = self._default
        return LLMResponse(
            content=content,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )


# ---------------------------------------------------------------------------
# Mock RAG engine
# ---------------------------------------------------------------------------


class _RecordingMockRagEngine:
    """Mock RAGEngine that records retrieve() calls and returns canned hits.

    Uses the same public surface as ``RAGEngine`` for the methods
    ``MetacouplingAssistant`` actually calls in pre_retrieval mode:
    ``retrieve(query, top_k, min_score)`` and the ``backend`` attribute.
    """

    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        backend: str = "embeddings",
    ) -> None:
        self._results = list(results or [])
        self.backend = backend
        self.total_chunks = max(len(self._results), 1)
        self.calls: list[dict[str, Any]] = []
        self.raise_on_retrieve: Exception | None = None

    def load(self) -> None:  # pragma: no cover - no-op for tests
        return None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        max_chunks_per_paper: int = 3,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
                "max_chunks_per_paper": max_chunks_per_paper,
            }
        )
        if self.raise_on_retrieve is not None:
            raise self.raise_on_retrieve
        return list(self._results[:top_k])


# ---------------------------------------------------------------------------
# Reusable fake passages
# ---------------------------------------------------------------------------


def _make_chunk(
    paper_key: str,
    title: str,
    authors: str,
    year: int,
    section: str,
    text: str,
    chunk_index: int = 0,
) -> TextChunk:
    return TextChunk(
        paper_key=paper_key,
        paper_title=title,
        authors=authors,
        year=year,
        section=section,
        text=text,
        chunk_index=chunk_index,
    )


@pytest.fixture
def fake_retrieval_results() -> list[RetrievalResult]:
    """Five hand-crafted RetrievalResult objects for prompt-injection tests."""
    return [
        RetrievalResult(
            chunk=_make_chunk(
                "liu_framing_2013",
                "Framing Sustainability in a Telecoupled World",
                "Liu, Jianguo and Hull, Vanessa",
                2013,
                "Introduction",
                "Telecoupling refers to socioeconomic and environmental "
                "interactions over distances, integrating coupled human "
                "and natural systems separated by large distances.",
            ),
            score=0.92,
        ),
        RetrievalResult(
            chunk=_make_chunk(
                "liu_integration_2017",
                "Integration across a metacoupled world",
                "Liu, Jianguo",
                2017,
                "Methods",
                "Metacoupling encompasses intracoupling within a system, "
                "pericoupling between adjacent systems, and telecoupling "
                "between distant systems.",
            ),
            score=0.87,
        ),
        RetrievalResult(
            chunk=_make_chunk(
                "sun_telecoupled_2017",
                "Telecoupled land-use changes in distant countries",
                "Sun, Jian and Tong, Yuxing",
                2017,
                "Results",
                "Soybean trade between Brazil and China drove substantial "
                "land-use change in Mato Grosso, with cropland expanding "
                "into Cerrado savanna ecosystems.",
            ),
            score=0.81,
        ),
        RetrievalResult(
            chunk=_make_chunk(
                "bicudo_sino_2017",
                "The Sino-Brazilian Telecoupled Soybean System",
                "Bicudo da Silva, Ramon and Milhorance, Carolina",
                2017,
                "Discussion",
                "Brazilian soybean exports to China grew rapidly after "
                "2003, restructuring agricultural landscapes in major "
                "producing states.",
            ),
            score=0.76,
        ),
        RetrievalResult(
            chunk=_make_chunk(
                "carlson_telecoupling_2017",
                "The Telecoupling Framework for Fisheries Management",
                "Carlson, Andrew and Taylor, William",
                2017,
                "Conclusion",
                "Spillover systems are often overlooked in telecoupling "
                "analyses but can experience effects of comparable or "
                "larger magnitude than sending or receiving systems.",
            ),
            score=0.68,
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client() -> _RecordingMockLLMClient:
    """A mock LLM client with empty default response.

    Tests that need a specific response should construct their own
    instance of ``_RecordingMockLLMClient`` with ``responses=[...]``.
    """
    return _RecordingMockLLMClient(default_response="### 1. Coupling Classification\nTest.")


@pytest.fixture
def mock_rag_engine(fake_retrieval_results) -> _RecordingMockRagEngine:
    """A mock RAG engine pre-loaded with five fake passages."""
    return _RecordingMockRagEngine(
        results=fake_retrieval_results, backend="embeddings"
    )


@pytest.fixture
def advisor_pre_retrieval(
    mock_llm_client: _RecordingMockLLMClient,
    mock_rag_engine: _RecordingMockRagEngine,
) -> MetacouplingAssistant:
    """A MetacouplingAssistant in pre_retrieval mode with the mock engine injected."""
    advisor = MetacouplingAssistant(
        llm_client=mock_llm_client,
        max_examples=0,
        verbose=False,
        rag_mode="pre_retrieval",
    )
    # Inject the mock engine directly — bypasses the file-system loader.
    advisor._rag_engine = mock_rag_engine
    return advisor


@pytest.fixture
def advisor_post_hoc(
    mock_llm_client: _RecordingMockLLMClient,
    mock_rag_engine: _RecordingMockRagEngine,
) -> MetacouplingAssistant:
    """A MetacouplingAssistant in post_hoc mode with the mock engine injected."""
    advisor = MetacouplingAssistant(
        llm_client=mock_llm_client,
        max_examples=0,
        verbose=False,
        rag_mode="post_hoc",
    )
    advisor._rag_engine = mock_rag_engine
    return advisor
