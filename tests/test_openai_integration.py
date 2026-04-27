"""Integration tests that make REAL OpenAI API calls.

These tests require a valid ``OPENAI_API_KEY`` environment variable or
a ``.env`` file in the project root.  They are **automatically skipped**
when no key is available, so ``pytest`` always passes in CI without
credentials.

Run explicitly::

    pytest tests/test_openai_integration.py -v
"""

from __future__ import annotations

import os
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Load .env if present (no external dependency required)
# ---------------------------------------------------------------------------

_ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"

if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_API_KEY = os.environ.get("OPENAI_API_KEY", "")

skip_no_key = pytest.mark.skipif(
    not _API_KEY,
    reason="OPENAI_API_KEY not set — skipping live API tests",
)


# ---------------------------------------------------------------------------
# Helper: build an adapter for a given model
# ---------------------------------------------------------------------------

def _make_adapter(model: str):
    from openai import OpenAI

    from metacouplingllm.llm.client import OpenAIAdapter

    client = OpenAI(api_key=_API_KEY)
    return OpenAIAdapter(client, model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@skip_no_key
class TestOpenAIAdapterGPT4oMini:
    """Smoke tests against gpt-4o-mini (old-style max_tokens)."""

    MODEL = "gpt-4o-mini"

    def test_basic_chat(self):
        from metacouplingllm.llm.client import Message

        adapter = _make_adapter(self.MODEL)
        resp = adapter.chat(
            messages=[Message(role="user", content="Say 'hello' and nothing else.")],
            temperature=0.0,
            max_tokens=10,
        )
        assert resp.content.strip().lower().startswith("hello")
        assert resp.usage.get("total_tokens", 0) > 0

    def test_max_tokens_respected(self):
        from metacouplingllm.llm.client import Message

        adapter = _make_adapter(self.MODEL)
        resp = adapter.chat(
            messages=[
                Message(role="user", content="Count from 1 to 1000."),
            ],
            max_tokens=20,
        )
        # Should be cut short
        assert len(resp.content) < 500


@skip_no_key
class TestOpenAIAdapterGPT5:
    """Smoke tests against gpt-5.4 (requires max_completion_tokens)."""

    MODEL = "gpt-5.4"

    def test_basic_chat(self):
        """Verify that the adapter handles gpt-5.4 correctly."""
        from metacouplingllm.llm.client import Message

        adapter = _make_adapter(self.MODEL)
        resp = adapter.chat(
            messages=[Message(role="user", content="Say 'hello' and nothing else.")],
            max_tokens=50,  # adapter must convert to max_completion_tokens
        )
        assert "hello" in resp.content.strip().lower()

    def test_max_tokens_retry_works(self):
        """The max_tokens → max_completion_tokens retry must succeed."""
        from metacouplingllm.llm.client import Message

        adapter = _make_adapter(self.MODEL)
        resp = adapter.chat(
            messages=[
                Message(
                    role="user",
                    content="Return the JSON: {\"status\": \"ok\"}",
                ),
            ],
            max_tokens=100,
        )
        assert "ok" in resp.content

    def test_structured_extraction_prompt(self):
        """Simulate the map extraction prompt that was failing."""
        from metacouplingllm.llm.client import Message

        adapter = _make_adapter(self.MODEL)
        resp = adapter.chat(
            messages=[
                Message(
                    role="system",
                    content=(
                        "You extract structured map data. "
                        "Return ONLY a JSON object."
                    ),
                ),
                Message(
                    role="user",
                    content=(
                        "Extract map data from this analysis:\n\n"
                        "Sending system: Brazil\n"
                        "Receiving system: China\n"
                        "Flow: matter, Brazil → China (soybeans)\n\n"
                        "Return JSON with focal_country, "
                        "receiving_countries, flows."
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=800,  # This is exactly what was failing
        )
        assert "BRA" in resp.content or "Brazil" in resp.content


@skip_no_key
class TestFullAdvisorIntegration:
    """End-to-end test: MetacouplingAssistant with real API calls."""

    def test_analyze_produces_map_data(self):
        """Full analysis with map extraction should populate map_data."""
        from metacouplingllm.core import MetacouplingAssistant

        adapter = _make_adapter("gpt-4o-mini")
        advisor = MetacouplingAssistant(
            llm_client=adapter,
            auto_map=True,
            verbose=True,
        )

        result = advisor.analyze(
            "Brazil's soybean exports to China."
        )

        # The analysis should produce text
        assert result.formatted
        assert result.parsed.is_parsed

        # The second LLM call should have extracted map data
        md = result.parsed.map_data
        assert md is not None, "map_data should be populated by second LLM call"
        assert md.get("focal_country"), "focal_country should be resolved"
        print(f"map_data: {md}")
