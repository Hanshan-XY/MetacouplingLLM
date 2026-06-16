"""Tests for built-in run tracing (metacouplingllm.tracing + MetacouplingAssistant)."""

from __future__ import annotations

import pytest

from metacouplingllm import MetacouplingAssistant, RunTrace
from metacouplingllm.llm.client import LLMResponse, Message
from metacouplingllm.tracing import CallRecord, _norm_usage, _sum_usage


_ANALYSIS = "### 1. Coupling Classification\nTelecoupling between systems.\n"


class _StubClient:
    """Minimal LLMClient returning a parseable analysis with usage."""

    model = "stub-model"

    def __init__(self, tool_uses=None, record_kwargs=None):
        self._tool_uses = tool_uses
        self._record_kwargs = record_kwargs  # mutable list to capture kwargs

    def chat(self, messages, temperature=0.7, max_tokens=None, **kwargs):
        if self._record_kwargs is not None:
            self._record_kwargs.append(kwargs)
        return LLMResponse(
            content=_ANALYSIS,
            usage={"prompt_tokens": 10, "completion_tokens": 5,
                   "total_tokens": 15},
            tool_uses=self._tool_uses,
        )


def _assistant(**kw):
    kw.setdefault("max_examples", 0)
    kw.setdefault("generate_abstract", False)
    return MetacouplingAssistant(_StubClient(), **kw)


# ---------------------------------------------------------------------------
# 1. analyze() writes turn1/ and attaches a RunTrace
# ---------------------------------------------------------------------------
def test_analyze_writes_turn1_and_attaches_runtrace(tmp_path):
    adv = _assistant(trace=True, trace_dir=tmp_path)
    result = adv.analyze("Brazil China soybean telecoupling")

    assert isinstance(result.trace, RunTrace)
    out = result.trace.out_dir
    assert out is not None and out.name == "turn1"
    names = {p.name for p in out.iterdir()}
    for expected in ("00_run_config.md", "05_llm_call_main_analysis.md",
                     "06_parsed_analysis.md", "09_formatted_output.md",
                     "10_pipeline_metadata.md", "README.md"):
        assert expected in names, expected
    assert result.trace.total_input_tokens > 0
    assert result.trace.total_output_tokens > 0
    assert any(c.label == "main_analysis" for c in result.trace.calls)


# ---------------------------------------------------------------------------
# 2. trace=False writes nothing and attaches no trace
# ---------------------------------------------------------------------------
def test_trace_false_writes_nothing(tmp_path):
    adv = _assistant(trace=False, trace_dir=tmp_path)
    result = adv.analyze("Brazil China soybean telecoupling")
    assert result.trace is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 3. default-off under the autouse fixture (no trace arg → None)
# ---------------------------------------------------------------------------
def test_default_off_under_autouse_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so any stray runs/ would land here
    adv = _assistant()  # trace=None → autouse flipped default to False
    result = adv.analyze("Brazil China soybean telecoupling")
    assert result.trace is None
    assert not (tmp_path / "runs").exists()


# ---------------------------------------------------------------------------
# 4. env-var kill switch disables even when the module default is on
# ---------------------------------------------------------------------------
def test_env_var_disable(tmp_path, monkeypatch):
    import metacouplingllm.tracing as _t
    monkeypatch.setattr(_t, "_DEFAULT_TRACE", True)
    monkeypatch.setenv("METACOUPLINGLLM_DISABLE_TRACE", "1")
    adv = _assistant(trace_dir=tmp_path)  # trace=None
    result = adv.analyze("Brazil China soybean telecoupling")
    assert result.trace is None
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 5. multi-turn writes turn1/ and turn2/ under one session root
# ---------------------------------------------------------------------------
def test_multi_turn_subdirs(tmp_path):
    adv = _assistant(trace=True, trace_dir=tmp_path)
    adv.analyze("Brazil China soybean telecoupling")
    adv.refine("Add the spillover systems")
    assert (tmp_path / "turn1").is_dir()
    assert (tmp_path / "turn2").is_dir()


# ---------------------------------------------------------------------------
# 6. token aggregation across both provider usage schemas
# ---------------------------------------------------------------------------
def test_sum_usage_handles_both_schemas():
    def _rec(usage):
        return CallRecord(
            n=1, label="x", messages=[], requested_temperature=0.7,
            max_tokens=None, extra_kwargs={}, response_content="",
            response_tool_uses=None, usage=usage, duration_s=0.0,
        )

    assert _norm_usage({"prompt_tokens": 3, "completion_tokens": 4}) == (3, 4, 7)
    assert _norm_usage({"input_tokens": 8, "output_tokens": 2}) == (8, 2, 10)
    calls = [
        _rec({"prompt_tokens": 3, "completion_tokens": 4}),       # OpenAI/Grok
        _rec({"input_tokens": 8, "output_tokens": 2}),            # Anthropic/Gemini
    ]
    assert _sum_usage(calls) == (11, 6)


# ---------------------------------------------------------------------------
# 7. proxy forwards **kwargs and captures tool_uses
# ---------------------------------------------------------------------------
def test_proxy_forwards_kwargs_and_captures_tooluses():
    from metacouplingllm.tracing import _RecordingClient
    received: list[dict] = []
    inner = _StubClient(tool_uses=[{"name": "submit"}], record_kwargs=received)
    proxy = _RecordingClient(inner)
    with proxy.label("web_extraction"):
        proxy.chat([Message(role="user", content="hi")],
                   temperature=0.0, response_format={"type": "json_object"})
    assert received and received[0].get("response_format") == {"type": "json_object"}
    rec = proxy.captured_calls[0]
    assert rec.label == "web_extraction"
    assert rec.response_tool_uses == [{"name": "submit"}]
    assert rec.extra_kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# 8. wrapping preserves adapter-type dispatch (web-search auto-wiring)
# ---------------------------------------------------------------------------
def test_proxy_preserves_adapter_dispatch():
    from metacouplingllm import OpenAIAdapter
    inner = OpenAIAdapter(client=object(), model="gpt-x")
    adv = _build_with_inner(inner, trace=True)
    # The proxy is NOT an adapter subclass; the unwrapped inner is.
    assert not isinstance(adv._client, OpenAIAdapter)
    assert isinstance(adv._client_inner, OpenAIAdapter)
    # Attribute delegation still works through the proxy.
    assert adv._client.model == "gpt-x"


def _build_with_inner(inner, **kw):
    kw.setdefault("max_examples", 0)
    kw.setdefault("generate_abstract", False)
    return MetacouplingAssistant(inner, **kw)


# ---------------------------------------------------------------------------
# 9. RAG-only mode traces (RAGResult.trace) without crashing
# ---------------------------------------------------------------------------
def test_rag_only_mode_traces(tmp_path):
    adv = _assistant(coupling_analysis=False, trace=True, trace_dir=tmp_path)
    result = adv.analyze("What does the literature say about soy telecoupling?")
    assert isinstance(result.trace, RunTrace)
    assert result.trace.out_dir is not None
    labels = {c.label for c in result.trace.calls}
    assert "rag_qa" in labels


# ---------------------------------------------------------------------------
# 10. artifact-write failure never breaks analyze(); out_dir → None
# ---------------------------------------------------------------------------
def test_graceful_failure_does_not_break_analyze(tmp_path, monkeypatch):
    import metacouplingllm.tracing as _t

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(_t, "write_run_artifacts", _boom)
    adv = _assistant(trace=True, trace_dir=tmp_path)
    result = adv.analyze("Brazil China soybean telecoupling")
    # Analysis still succeeds; trace attached but out_dir is None.
    assert result.formatted
    assert isinstance(result.trace, RunTrace)
    assert result.trace.out_dir is None
