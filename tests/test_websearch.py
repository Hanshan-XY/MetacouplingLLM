"""Tests for knowledge/websearch.py — web search integration."""

from unittest.mock import patch

from metacouplingllm.knowledge.websearch import (
    _DuckDuckGoLiteParser,
    _build_search_queries,
    _extract_json_object,
    _resolve_ddg_url,
    _search_stdlib,
    AnthropicWebSearchBackend,
    OpenAIWebSearchBackend,
    annotate_web_citations,
    extract_web_map_signals,
    format_web_context,
    format_web_map_signals_context,
    search_web,
)
from metacouplingllm.prompts.builder import PromptBuilder


# ---------------------------------------------------------------------------
# format_web_context
# ---------------------------------------------------------------------------

class TestFormatWebContext:
    def test_empty_results(self):
        assert format_web_context([]) == ""

    def test_formats_results(self):
        results = [
            {
                "title": "Michigan Pork Exports Rise",
                "snippet": "Michigan exported $200M in pork last year.",
                "url": "https://example.com/pork",
            },
            {
                "title": "US Pork Trade Data",
                "snippet": "Top destinations include Japan and Mexico.",
                "url": "https://example.com/trade",
            },
        ]
        text = format_web_context(results)
        assert "WEB SEARCH CONTEXT" in text
        assert "Michigan Pork Exports Rise" in text
        assert "$200M" in text
        assert "Japan and Mexico" in text
        assert "https://example.com/pork" in text

    def test_uses_w_prefix_numbering(self):
        """Web results use [W1], [W2] to avoid collision with literature [1]."""
        results = [
            {"title": "A", "snippet": "s1", "url": "https://a.com"},
            {"title": "B", "snippet": "s2", "url": "https://b.com"},
        ]
        text = format_web_context(results)
        assert "[W1]" in text
        assert "[W2]" in text
        # Must NOT contain bare [1] or [2] numbering
        assert "\n1." not in text
        assert "\n2." not in text

    def test_handles_missing_fields(self):
        results = [{"title": "Test", "snippet": "", "url": ""}]
        text = format_web_context(results)
        assert "Test" in text
        assert "WEB SEARCH CONTEXT" in text


class TestStructuredWebMapSignals:
    def test_extract_web_map_signals_validates_json_output(self):
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.0, max_tokens=None):
                return LLMResponse(
                    content=(
                        '{"focal_country":"Brazil",'
                        '"receiving_systems":[{"country":"China","kind":"direct","confidence":0.91,"evidence":["W1"],"reason":"Explicit importer"}],'
                        '"spillover_systems":[{"country":"United States","kind":"proxy","confidence":0.75,"evidence":["W2"],"reason":"Competing exporter"}],'
                        '"flows":[{"category":"matter","source_country":"Brazil","target_country":"China","kind":"direct","confidence":0.93,"evidence":["W1"],"description":"Soybean exports"}]}'
                    )
                )

        results = [
            {
                "title": "Brazil soybean exports to China",
                "snippet": "China remains Brazil's largest soybean export market.",
                "url": "https://example.com/china",
            },
            {
                "title": "U.S. and Brazil soybean competitiveness",
                "snippet": "The United States competes with Brazil in soybean exports.",
                "url": "https://example.com/usa",
            },
        ]

        signals = extract_web_map_signals(
            "Brazil soybean production and exports",
            results,
            MockClient(),
            min_confidence=0.7,
        )

        assert signals is not None
        assert signals["focal_country"] == "BRA"
        assert signals["receiving_systems"][0]["country"] == "CHN"
        assert signals["spillover_systems"][0]["country"] == "USA"
        assert signals["flows"][0]["direction"] == "Brazil → China"

    def test_format_web_map_signals_context(self):
        context = format_web_map_signals_context(
            {
                "focal_country": "BRA",
                "receiving_systems": [
                    {
                        "country": "CHN",
                        "kind": "direct",
                        "confidence": 0.91,
                        "evidence": ["W1"],
                    }
                ],
                "spillover_systems": [],
                "flows": [
                    {
                        "category": "matter",
                        "direction": "Brazil → China",
                        "kind": "direct",
                        "confidence": 0.93,
                        "evidence": ["W1"],
                    }
                ],
            }
        )

        assert "STRUCTURED WEB MAP SIGNALS" in context
        assert "CHN" in context
        assert "Brazil → China" in context


# ---------------------------------------------------------------------------
# search_web (live test — only runs if network is available)
# ---------------------------------------------------------------------------

class TestOpenAIWebSearchBackend:
    def test_returns_normalized_results_and_omits_reasoning_when_default(self):
        class MockResponse:
            output_text = (
                '{"results": ['
                '{"title": "Brazil soybean exports", '
                '"url": "https://example.com/brazil", '
                '"snippet": "Brazil exports soybeans to China."}'
                "]}"
            )

            def model_dump(self):
                return {"output": []}

        class MockResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.responses = MockResponses()

        client = MockClient()
        backend = OpenAIWebSearchBackend(client=client, model="gpt-5")
        results = backend.search("Brazil soybean exports", max_results=3)

        assert results == [{
            "title": "Brazil soybean exports",
            "url": "https://example.com/brazil",
            "snippet": "Brazil exports soybeans to China.",
        }]
        assert client.responses.kwargs["model"] == "gpt-5"
        assert client.responses.kwargs["tools"] == [{
            "type": "web_search",
            "external_web_access": True,
        }]
        assert "reasoning" not in client.responses.kwargs

    def test_falls_back_to_sources_when_json_missing(self):
        class MockResponse:
            output_text = "not json"

            def model_dump(self):
                return {
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {
                                "sources": [
                                    {
                                        "title": "Namibia fisheries report",
                                        "url": "https://example.com/namibia",
                                        "snippet": "Grounded source snippet.",
                                    }
                                ]
                            },
                        }
                    ]
                }

        class MockResponses:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.responses = MockResponses()

        backend = OpenAIWebSearchBackend(client=MockClient(), model="gpt-5")
        results = backend.search("Namibia fisheries sustainability", max_results=2)

        assert results == [{
            "title": "Namibia fisheries report",
            "url": "https://example.com/namibia",
            "snippet": "Grounded source snippet.",
        }]


class TestAnthropicWebSearchBackend:
    def test_extracts_results_from_citations_primary_path(self):
        """Happy path: citations on text blocks give url + title + cited_text."""

        class MockResponse:
            def model_dump(self):
                return {
                    "content": [
                        {"type": "text", "text": "I'll search for this."},
                        {
                            "type": "server_tool_use",
                            "id": "srvtoolu_1",
                            "name": "web_search",
                            "input": {"query": "Brazil soybean exports"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/brazil",
                                    "title": "Brazil soybean exports",
                                    "encrypted_content": "opaque",
                                    "page_age": "April 12, 2026",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": "Brazil remains the top soybean exporter.",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/brazil",
                                    "title": "Brazil soybean exports",
                                    "encrypted_index": "idx",
                                    "cited_text": "Brazil exported 100M tons of soybeans in 2025.",
                                }
                            ],
                        },
                    ]
                }

        class MockMessages:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        client = MockClient()
        backend = AnthropicWebSearchBackend(
            client=client, model="claude-opus-4-7", max_uses=3,
        )
        results = backend.search("Brazil soybean exports", max_results=3)

        assert results == [{
            "title": "Brazil soybean exports",
            "url": "https://example.com/brazil",
            "snippet": "Brazil exported 100M tons of soybeans in 2025.",
        }]
        # Verify the tool was configured correctly
        assert client.messages.kwargs["model"] == "claude-opus-4-7"
        assert client.messages.kwargs["tools"] == [{
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": 3,
        }]

    def test_falls_back_to_tool_result_when_no_citations(self):
        """Fallback: no citations → walk web_search_tool_result blocks."""

        class MockResponse:
            def model_dump(self):
                return {
                    "content": [
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": [
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/report",
                                    "title": "Fisheries Report",
                                    "encrypted_content": "opaque",
                                    "page_age": "March 1, 2026",
                                },
                                {
                                    "type": "web_search_result",
                                    "url": "https://example.com/analysis",
                                    "title": "Deep Analysis",
                                    "encrypted_content": "opaque",
                                    "page_age": "March 8, 2026",
                                },
                            ],
                        },
                        {"type": "text", "text": "No citations here."},
                    ]
                }

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        backend = AnthropicWebSearchBackend(client=MockClient())
        results = backend.search("fisheries report", max_results=2)

        assert results == [
            {
                "title": "Fisheries Report",
                "url": "https://example.com/report",
                "snippet": "(page age: March 1, 2026)",
            },
            {
                "title": "Deep Analysis",
                "url": "https://example.com/analysis",
                "snippet": "(page age: March 8, 2026)",
            },
        ]

    def test_dedupes_repeated_url_across_citations(self):
        """Same URL cited twice → only the first citation's snippet is kept."""

        class MockResponse:
            def model_dump(self):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Fact one.",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/a",
                                    "title": "A",
                                    "cited_text": "First passage.",
                                }
                            ],
                        },
                        {
                            "type": "text",
                            "text": "Fact two.",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/a",
                                    "title": "A",
                                    "cited_text": "Second passage — should be discarded.",
                                },
                                {
                                    "type": "web_search_result_location",
                                    "url": "https://example.com/b",
                                    "title": "B",
                                    "cited_text": "Unique snippet.",
                                },
                            ],
                        },
                    ]
                }

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        backend = AnthropicWebSearchBackend(client=MockClient())
        results = backend.search("test", max_results=5)

        assert results == [
            {"title": "A", "url": "https://example.com/a", "snippet": "First passage."},
            {"title": "B", "url": "https://example.com/b", "snippet": "Unique snippet."},
        ]

    def test_respects_max_results_from_citations(self):
        """max_results caps the return even if more citations exist."""

        class MockResponse:
            def model_dump(self):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": "Many facts.",
                            "citations": [
                                {
                                    "type": "web_search_result_location",
                                    "url": f"https://example.com/{i}",
                                    "title": f"Source {i}",
                                    "cited_text": f"Snippet {i}.",
                                }
                                for i in range(10)
                            ],
                        }
                    ]
                }

        class MockMessages:
            def create(self, **kwargs):
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        backend = AnthropicWebSearchBackend(client=MockClient())
        results = backend.search("many results", max_results=3)

        assert len(results) == 3
        assert [r["url"] for r in results] == [
            "https://example.com/0",
            "https://example.com/1",
            "https://example.com/2",
        ]

    def test_returns_empty_for_empty_query(self):
        class MockClient:
            pass

        backend = AnthropicWebSearchBackend(client=MockClient())
        assert backend.search("", max_results=5) == []
        assert backend.search("   ", max_results=5) == []

    def test_passes_optional_tool_params_when_configured(self):
        """allowed_domains, blocked_domains, user_location flow through."""

        captured: dict[str, object] = {}

        class MockResponse:
            def model_dump(self):
                return {"content": []}

        class MockMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        backend = AnthropicWebSearchBackend(
            client=MockClient(),
            allowed_domains=["example.com"],
            blocked_domains=["spam.com"],
            user_location={"type": "approximate", "country": "US"},
            tool_version="web_search_20250305",
        )
        backend.search("anything", max_results=1)

        tool = captured["tools"][0]
        assert tool["type"] == "web_search_20250305"
        assert tool["allowed_domains"] == ["example.com"]
        assert tool["blocked_domains"] == ["spam.com"]
        assert tool["user_location"] == {"type": "approximate", "country": "US"}


class TestAnthropicWebSearchBackendToolVersionAutoSelect:
    """Tests for auto-selection of tool_version based on model."""

    @staticmethod
    def _capturing_client():
        captured: dict[str, object] = {}

        class MockResponse:
            def model_dump(self):
                return {"content": []}

        class MockMessages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return MockResponse()

        class MockClient:
            def __init__(self):
                self.messages = MockMessages()

        return MockClient(), captured

    def test_auto_selects_20260209_for_opus_4_7(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-opus-4-7").search("x")
        assert captured["tools"][0]["type"] == "web_search_20260209"

    def test_auto_selects_20260209_for_opus_4_6(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-opus-4-6").search("x")
        assert captured["tools"][0]["type"] == "web_search_20260209"

    def test_auto_selects_20260209_for_sonnet_4_6(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-sonnet-4-6").search("x")
        assert captured["tools"][0]["type"] == "web_search_20260209"

    def test_auto_selects_20250305_for_older_opus(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-opus-4-5").search("x")
        assert captured["tools"][0]["type"] == "web_search_20250305"

    def test_auto_selects_20250305_for_older_sonnet(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-sonnet-4-5").search("x")
        assert captured["tools"][0]["type"] == "web_search_20250305"

    def test_auto_selects_20250305_for_haiku(self):
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(client=client, model="claude-haiku-4-5").search("x")
        assert captured["tools"][0]["type"] == "web_search_20250305"

    def test_unknown_model_defaults_to_newer_version(self):
        """Forward-compat: unknown model IDs get the newer tool version."""
        client, captured = self._capturing_client()
        AnthropicWebSearchBackend(
            client=client, model="claude-opus-5-0-hypothetical",
        ).search("x")
        assert captured["tools"][0]["type"] == "web_search_20260209"

    def test_explicit_tool_version_overrides_auto(self):
        """Explicit tool_version wins over the model-based lookup."""
        client, captured = self._capturing_client()
        # Newer model, but explicitly request the older tool version.
        AnthropicWebSearchBackend(
            client=client,
            model="claude-opus-4-7",
            tool_version="web_search_20250305",
        ).search("x")
        assert captured["tools"][0]["type"] == "web_search_20250305"


class TestSearchWeb:
    def test_returns_list(self):
        results = search_web("metacoupling framework", max_results=2)
        # May return empty if no network; just check the type
        assert isinstance(results, list)

    def test_results_have_expected_keys(self):
        results = search_web("telecoupling soybean trade", max_results=2)
        for r in results:
            assert "title" in r
            assert "snippet" in r
            assert "url" in r

    def test_respects_max_results(self):
        results = search_web("telecoupling", max_results=3)
        assert len(results) <= 3

    def test_uses_custom_backend_before_builtin_search(self):
        class StubBackend:
            def search(self, query, max_results=5):
                return [{
                    "title": f"Result for {query}",
                    "snippet": "stub snippet",
                    "url": "https://example.com/stub",
                }]

        with patch("metacouplingllm.knowledge.websearch._build_search_queries") as build_queries:
            results = search_web(
                "telecoupling",
                max_results=3,
                backend=StubBackend(),
            )

        assert results == [{
            "title": "Result for telecoupling",
            "snippet": "stub snippet",
            "url": "https://example.com/stub",
        }]
        build_queries.assert_not_called()


# ---------------------------------------------------------------------------
# Stdlib fallback — _DuckDuckGoLiteParser
# ---------------------------------------------------------------------------

SAMPLE_LITE_HTML = """
<table>
<tr>
  <td>1.</td>
  <td><a rel="nofollow" class="result-link" href="https://example.com/page1">
    First Result Title
  </a></td>
</tr>
<tr>
  <td></td>
  <td class="result-snippet">This is the first snippet.</td>
</tr>
<tr>
  <td>2.</td>
  <td><a rel="nofollow" class="result-link" href="https://example.com/page2">
    Second Result Title
  </a></td>
</tr>
<tr>
  <td></td>
  <td class="result-snippet">This is the second snippet.</td>
</tr>
</table>
"""


class TestDuckDuckGoLiteParser:
    def test_parses_results(self):
        parser = _DuckDuckGoLiteParser()
        parser.feed(SAMPLE_LITE_HTML)
        assert len(parser.results) == 2
        assert parser.results[0]["title"] == "First Result Title"
        assert parser.results[0]["snippet"] == "This is the first snippet."
        assert parser.results[0]["url"] == "https://example.com/page1"
        assert parser.results[1]["title"] == "Second Result Title"

    def test_empty_html(self):
        parser = _DuckDuckGoLiteParser()
        parser.feed("<html><body>No results</body></html>")
        assert parser.results == []

    def test_skips_link_without_url(self):
        html = """
        <a rel="nofollow" class="result-link" href="">No URL</a>
        <td class="result-snippet">Snippet for missing URL.</td>
        """
        parser = _DuckDuckGoLiteParser()
        parser.feed(html)
        assert parser.results == []


class TestResolveDdgUrl:
    def test_resolves_redirect_url(self):
        raw = (
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fnppc.org%2Freport.pdf"
            "&rut=abc123"
        )
        assert _resolve_ddg_url(raw) == "https://nppc.org/report.pdf"

    def test_preserves_normal_url(self):
        assert _resolve_ddg_url("https://example.com/page") == "https://example.com/page"

    def test_handles_empty_string(self):
        assert _resolve_ddg_url("") == ""

    def test_resolves_full_https_redirect(self):
        raw = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=x"
        assert _resolve_ddg_url(raw) == "https://example.com"


class TestAnnotateWebCitations:
    WEB_RESULTS = [
        {
            "title": "Michigan Pork Industry Report",
            "snippet": "Michigan pork production contributes $874 million GDP",
            "url": "https://example.com/pork",
        },
        {
            "title": "US Pork Exports Data",
            "snippet": "United States exported pork to Japan Mexico Korea",
            "url": "https://example.com/trade",
        },
    ]

    def test_adds_web_citations(self):
        text = (
            "Michigan pork production contributes significantly to GDP.\n"
            "The industry is important."
        )
        result = annotate_web_citations(text, self.WEB_RESULTS)
        assert "[W1]" in result

    def test_does_not_annotate_empty_results(self):
        text = "Some analysis line."
        result = annotate_web_citations(text, [])
        assert result == text

    def test_skips_section_headers(self):
        text = "COUPLING CLASSIFICATION\nSome pork production line."
        result = annotate_web_citations(text, self.WEB_RESULTS)
        lines = result.split("\n")
        assert "[W" not in lines[0]

    def test_skips_evidence_block(self):
        text = (
            "Michigan pork production contributes GDP.\n"
            "==============================\n"
            "  SUPPORTING EVIDENCE FROM LITERATURE\n"
            "==============================\n"
            "Michigan pork production contributes GDP."
        )
        result = annotate_web_citations(text, self.WEB_RESULTS)
        # Only the first line (before the evidence block) should be annotated
        lines = result.split("\n")
        assert "[W" in lines[0]
        assert "[W" not in lines[-1]

    def test_avoids_duplicate_citations(self):
        text = "Michigan pork production contributes GDP [W1]"
        result = annotate_web_citations(text, self.WEB_RESULTS)
        # Should not add another [W1]
        assert result.count("[W1]") == 1


class TestStdlibFallback:
    def test_falls_back_to_stdlib(self):
        """search_web falls back to stdlib when ddgs/duckduckgo_search absent."""
        fake_results = [
            {"title": "Fallback", "snippet": "Works", "url": "https://x.com"},
        ]
        with patch(
            "metacouplingllm.knowledge.websearch._search_stdlib",
            return_value=fake_results,
        ) as mock_stdlib:
            # Simulate both libraries missing
            with patch.dict("sys.modules", {"ddgs": None, "duckduckgo_search": None}):
                results = search_web("test query", max_results=1)
        assert results == fake_results

    def test_stdlib_respects_max_results(self):
        many = [{"title": f"R{i}", "snippet": "", "url": f"https://x.com/{i}"}
                for i in range(10)]
        with patch(
            "metacouplingllm.knowledge.websearch._search_stdlib",
            return_value=many[:3],
        ):
            with patch.dict("sys.modules", {"ddgs": None, "duckduckgo_search": None}):
                results = search_web("test", max_results=3)
        assert len(results) <= 3


class TestSearchBackendMerging:
    def test_merges_unique_results_across_backends(self):
        ddgs_results = [
            {"title": "A", "snippet": "s1", "url": "https://a.com"},
            {"title": "B", "snippet": "s2", "url": "https://b.com"},
        ]
        legacy_results = [
            {"title": "B duplicate", "snippet": "s2", "url": "https://b.com"},
            {"title": "C", "snippet": "s3", "url": "https://c.com"},
        ]
        stdlib_results = [
            {"title": "D", "snippet": "s4", "url": "https://d.com"},
        ]
        with patch(
            "metacouplingllm.knowledge.websearch._search_ddgs",
            return_value=ddgs_results,
        ), patch(
            "metacouplingllm.knowledge.websearch._search_duckduckgo_search",
            return_value=legacy_results,
        ), patch(
            "metacouplingllm.knowledge.websearch._search_stdlib",
            return_value=stdlib_results,
        ):
            results = search_web("test", max_results=4)

        assert len(results) == 4
        assert [r["url"] for r in results] == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
            "https://d.com",
        ]

    def test_stops_after_reaching_requested_count(self):
        ddgs_results = [
            {"title": "A", "snippet": "s1", "url": "https://a.com"},
            {"title": "B", "snippet": "s2", "url": "https://b.com"},
            {"title": "C", "snippet": "s3", "url": "https://c.com"},
        ]
        with patch(
            "metacouplingllm.knowledge.websearch._search_ddgs",
            return_value=ddgs_results,
        ), patch(
            "metacouplingllm.knowledge.websearch._search_duckduckgo_search",
        ) as mock_legacy, patch(
            "metacouplingllm.knowledge.websearch._search_stdlib",
        ) as mock_stdlib:
            results = search_web("test", max_results=2)

        assert len(results) == 2
        mock_legacy.assert_not_called()
        mock_stdlib.assert_not_called()


class TestSearchWebMetadata:
    """Tests for the optional metadata out-dict on search_web()."""

    def test_records_anthropic_backend_on_success(self):
        class StubAnthropic:
            model = "claude-opus-4-7"

            def search(self, query, max_results=5):
                return [{"title": "A", "snippet": "s", "url": "https://a.com"}]

        # Make isinstance(..., AnthropicWebSearchBackend) hit by patching
        # the class check, OR construct a real backend instance with a
        # mock client. Simpler: build a real AnthropicWebSearchBackend
        # subclass that overrides search().
        class RealishBackend(AnthropicWebSearchBackend):
            def search(self, query, max_results=5):
                return [{"title": "A", "snippet": "s", "url": "https://a.com"}]

        backend = RealishBackend(client=None, model="claude-opus-4-7")
        meta: dict[str, object] = {}
        results = search_web("anything", max_results=5, backend=backend, metadata=meta)
        assert len(results) == 1
        assert meta["backend_used"] == "Claude web_search (claude-opus-4-7)"
        assert "fallback_from" not in meta

    def test_records_openai_backend_on_success(self):
        class RealishBackend(OpenAIWebSearchBackend):
            def search(self, query, max_results=5):
                return [{"title": "B", "snippet": "s", "url": "https://b.com"}]

        backend = RealishBackend(client=None, model="gpt-5")
        meta: dict[str, object] = {}
        search_web("anything", max_results=5, backend=backend, metadata=meta)
        assert meta["backend_used"] == "OpenAI web_search (gpt-5)"
        assert "fallback_from" not in meta

    def test_records_duckduckgo_when_no_backend(self):
        meta: dict[str, object] = {}
        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            return_value=[
                {"title": "DDG", "snippet": "s", "url": "https://ddg.com"},
            ],
        ):
            search_web("metacoupling water", max_results=3, metadata=meta)
        assert meta["backend_used"] == "DuckDuckGo (fan-out + top-up)"
        assert "fallback_from" not in meta

    def test_records_fallback_after_backend_exception(self):
        class RaisingBackend(AnthropicWebSearchBackend):
            def search(self, query, max_results=5):
                raise RuntimeError("API down")

        backend = RaisingBackend(client=None, model="claude-opus-4-7")
        meta: dict[str, object] = {}
        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            return_value=[
                {"title": "DDG", "snippet": "s", "url": "https://ddg.com"},
            ],
        ):
            search_web(
                "metacoupling water", max_results=3,
                backend=backend, metadata=meta,
            )
        assert meta["backend_used"] == "DuckDuckGo (fan-out + top-up)"
        assert (
            meta["fallback_from"]
            == "Claude web_search (claude-opus-4-7) raised: API down"
        )

    def test_records_fallback_after_backend_empty(self):
        class EmptyBackend(OpenAIWebSearchBackend):
            def search(self, query, max_results=5):
                return []

        backend = EmptyBackend(client=None, model="gpt-5")
        meta: dict[str, object] = {}
        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            return_value=[
                {"title": "DDG", "snippet": "s", "url": "https://ddg.com"},
            ],
        ):
            search_web(
                "metacoupling water", max_results=3,
                backend=backend, metadata=meta,
            )
        assert meta["backend_used"] == "DuckDuckGo (fan-out + top-up)"
        assert (
            meta["fallback_from"]
            == "OpenAI web_search (gpt-5) returned 0 results"
        )

    def test_metadata_is_optional_for_backwards_compat(self):
        """search_web works the same when metadata is omitted (default)."""
        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            return_value=[
                {"title": "X", "snippet": "s", "url": "https://x.com"},
            ],
        ):
            results = search_web("metacoupling water", max_results=2)
        assert len(results) == 1


class TestStdlibNetworkErrorHandling:
    """Tests that _search_stdlib handles network failures gracefully."""

    def test_handles_url_error(self):
        """Network unavailable → returns empty list, no crash."""
        import urllib.error
        with patch(
            "metacouplingllm.knowledge.websearch.urllib.request.urlopen",
            side_effect=urllib.error.URLError("DNS failed"),
        ):
            result = _search_stdlib("test query", 5)
        assert result == []

    def test_handles_http_error(self):
        """HTTP 403 → returns empty list."""
        import urllib.error
        with patch(
            "metacouplingllm.knowledge.websearch.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://x.com", code=403, msg="Forbidden",
                hdrs=None, fp=None,  # type: ignore
            ),
        ):
            result = _search_stdlib("test", 5)
        assert result == []

    def test_handles_timeout(self):
        """Socket timeout → returns empty list."""
        import socket
        with patch(
            "metacouplingllm.knowledge.websearch.urllib.request.urlopen",
            side_effect=socket.timeout("Connection timed out"),
        ):
            result = _search_stdlib("test", 5)
        assert result == []

    def test_handles_malformed_html(self):
        """Malformed HTML that raises during parsing → returns empty list."""
        from io import BytesIO
        from unittest.mock import MagicMock

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><not>valid<xml"
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "metacouplingllm.knowledge.websearch.urllib.request.urlopen",
            return_value=mock_resp,
        ):
            # Even if HTML is malformed, parser should handle gracefully
            result = _search_stdlib("test", 5)
        assert isinstance(result, list)

    def test_truncates_long_queries(self):
        """Very long queries should be truncated to avoid URL length issues."""
        import urllib.error

        long_query = "x " * 2000  # 4000 chars
        captured_urls: list[str] = []

        original_urlopen = urllib.request.urlopen

        def mock_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            raise urllib.error.URLError("mock network error")

        with patch(
            "metacouplingllm.knowledge.websearch.urllib.request.urlopen",
            side_effect=mock_urlopen,
        ):
            result = _search_stdlib(long_query, 5)
        assert result == []
        # The query in the URL should be truncated
        assert len(captured_urls) == 1
        # URL-encoded query param should not contain the full 4000-char query
        assert len(captured_urls[0]) < 6500  # generous limit for URL encoding


class TestSearchQueryExpansion:
    def test_builds_partner_queries_for_subnational_export_topic(self):
        queries = _build_search_queries(
            "My research will analyze Michigan's pork production and exports."
        )
        assert queries[0] == "Michigan pork export destinations"
        assert "Michigan pork trade partners" in queries
        assert "United States pork exports top destinations" in queries
        assert queries[-1] == (
            "My research will analyze Michigan's pork production and exports."
        )

    def test_builds_queries_for_multiple_commodities(self):
        queries = _build_search_queries(
            "My research examines Michigan's pork and cherries exports."
        )
        assert "Michigan pork export destinations" in queries
        assert "Michigan cherries export destinations" in queries
        assert "United States cherries exports top destinations" in queries

    def test_national_level_trade_query_expands(self):
        """National trade queries should get expanded, not sent raw."""
        queries = _build_search_queries(
            "My research will analyze corn production and exports in USA"
        )
        # Should NOT send the raw query as the first item
        assert queries[0] != (
            "My research will analyze corn production and exports in USA"
        )
        # Should have expanded trade-specific queries
        found_destinations = any("destinations" in q.lower() for q in queries)
        found_partners = any("partners" in q.lower() or "markets" in q.lower() for q in queries)
        assert found_destinations, f"No destination query found in {queries}"
        assert found_partners, f"No partner/market query found in {queries}"

    def test_national_level_includes_raw_fallback(self):
        """National query expansion should include the raw query as fallback."""
        raw = "My research will analyze corn production and exports in USA"
        queries = _build_search_queries(raw)
        assert raw in queries, "Raw query should be included as fallback"

    def test_national_brazil_soybean_expands(self):
        """Brazil soybean export query should expand."""
        queries = _build_search_queries(
            "Brazil's soybean exports to China and beyond"
        )
        found_brazil = any("Brazil" in q for q in queries)
        assert found_brazil, f"Brazil not found in expanded queries: {queries}"

    def test_non_trade_prompt_keeps_raw_query(self):
        query = "My research examines Michigan water quality."
        assert _build_search_queries(query) == [query]

    def test_search_web_merges_results_across_expanded_queries(self):
        def fake_single_query(query: str, max_results: int):
            mapping = {
                "Michigan pork export destinations": [
                    {"title": "A", "snippet": "s1", "url": "https://a.com"},
                ],
                "Michigan pork trade partners": [
                    {"title": "B", "snippet": "s2", "url": "https://b.com"},
                ],
                "Michigan pork export markets": [
                    {"title": "C", "snippet": "s3", "url": "https://c.com"},
                ],
            }
            return mapping.get(query, [])[:max_results]

        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            side_effect=fake_single_query,
        ) as mock_single:
            results = search_web(
                "My research will analyze Michigan's pork production and exports.",
                max_results=3,
            )

        assert len(results) == 3
        assert [r["url"] for r in results] == [
            "https://a.com",
            "https://b.com",
            "https://c.com",
        ]
        assert mock_single.call_args_list[0].args[0] == "Michigan pork export destinations"

    def test_top_up_runs_raw_query_at_full_budget_when_fanout_short(self):
        """If fan-out's per-variant cap leaves us short, top-up re-issues
        the raw query at full budget and merges fresh URLs."""
        raw_query = (
            "My research will analyze Michigan's pork production and exports."
        )
        calls: list[tuple[str, int]] = []

        def fake_single_query(q: str, mr: int):
            calls.append((q, mr))
            # Variant calls (small budget): each returns 2 unique URLs
            if mr <= 6:
                idx = len(calls)
                return [
                    {"title": f"V{idx}_{i}", "snippet": "",
                     "url": f"https://variant.com/{idx}/{i}"}
                    for i in range(2)
                ]
            # Top-up call (full budget): returns 30 fresh URLs
            return [
                {"title": f"T{i}", "snippet": "",
                 "url": f"https://topup.com/{i}"}
                for i in range(30)
            ]

        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            side_effect=fake_single_query,
        ):
            results = search_web(raw_query, max_results=20)

        # Variant phase + top-up phase. Variant calls all use mr <= 6;
        # the final call must be the raw query at full budget.
        variant_calls = [c for c in calls if c[1] <= 6]
        topup_calls = [c for c in calls if c[1] == 20]
        assert len(variant_calls) >= 2, f"expected fan-out, saw calls={calls}"
        assert len(topup_calls) == 1, f"expected one top-up, saw calls={calls}"
        assert topup_calls[0][0] == raw_query

        # Final result reaches max_results once top-up fills in.
        assert len(results) == 20
        # At least some top-up URLs must appear in the final merged set.
        topup_urls_in_result = [
            r for r in results if r["url"].startswith("https://topup.com/")
        ]
        assert topup_urls_in_result, (
            f"top-up results not merged into final: {[r['url'] for r in results]}"
        )

    def test_top_up_skipped_when_fanout_already_full(self):
        """If fan-out already returned max_results unique URLs, no top-up."""
        raw_query = (
            "My research will analyze Michigan's pork production and exports."
        )
        calls: list[tuple[str, int]] = []

        def fake_single_query(q: str, mr: int):
            calls.append((q, mr))
            # Each variant returns 6 fresh URLs (per-variant cap saturates)
            idx = len(calls)
            return [
                {"title": f"V{idx}_{i}", "snippet": "",
                 "url": f"https://variant.com/{idx}/{i}"}
                for i in range(6)
            ]

        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            side_effect=fake_single_query,
        ):
            results = search_web(raw_query, max_results=10)

        # No top-up call should have happened: every call uses the small cap.
        topup_calls = [c for c in calls if c[1] == 10]
        assert topup_calls == [], f"unexpected top-up call: {calls}"
        assert len(results) == 10

    def test_top_up_skipped_for_non_trade_query(self):
        """Non-trade queries have only one variant; no top-up should fire
        even if the single call returns fewer than max_results."""
        calls: list[tuple[str, int]] = []

        def fake_single_query(q: str, mr: int):
            calls.append((q, mr))
            # Return only 3 URLs even though caller asked for 20
            return [
                {"title": f"R{i}", "snippet": "",
                 "url": f"https://r.com/{i}"}
                for i in range(3)
            ]

        with patch(
            "metacouplingllm.knowledge.websearch._search_single_query",
            side_effect=fake_single_query,
        ):
            results = search_web("Michigan water quality study", max_results=20)

        # Only one call (the single variant). No top-up — that would just be
        # a redundant call to the same query at the same max.
        assert len(calls) == 1, f"expected exactly one call, saw {calls}"
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Builder integration
# ---------------------------------------------------------------------------

class TestBuilderWebContext:
    def test_web_context_included_in_system_prompt(self):
        builder = PromptBuilder(max_examples=0)
        web_ctx = (
            "## WEB SEARCH CONTEXT\n\n"
            "1. **Michigan Pork Exports**\n"
            "   Michigan exports pork to 30 countries.\n"
        )
        prompt = builder.build_system_prompt(
            research_context="Michigan pork exports",
            web_context=web_ctx,
        )
        assert "WEB SEARCH CONTEXT" in prompt
        assert "Michigan exports pork to 30 countries" in prompt

    def test_no_web_context_by_default(self):
        builder = PromptBuilder(max_examples=0)
        prompt = builder.build_system_prompt(
            research_context="Michigan pork exports",
        )
        assert "WEB SEARCH CONTEXT" not in prompt


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------

class TestAdvisorWebSearch:
    def test_web_search_flag_stored(self):
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        advisor = MetacouplingAssistant(
            llm_client=MockClient(),
            web_search=True,
            web_search_max_results=3,
        )
        assert advisor._web_search is True
        assert advisor._web_search_max_results == 3

    def test_web_search_disabled_by_default(self):
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        advisor = MetacouplingAssistant(llm_client=MockClient())
        assert advisor._web_search is False

    def test_structured_web_signals_added_to_prompt_and_result(
        self, monkeypatch
    ):
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import LLMResponse

        class MockClient:
            def __init__(self):
                self.last_messages = None

            def chat(self, messages, temperature=0.7, max_tokens=None):
                self.last_messages = messages
                return LLMResponse(content="Test response.")

        fake_results = [
            {"title": "A", "snippet": "Brazil exports to China", "url": "https://a.com"},
        ]
        fake_signals = {
            "focal_country": "BRA",
            "receiving_systems": [
                {
                    "country": "CHN",
                    "kind": "direct",
                    "confidence": 0.91,
                    "evidence": ["W1"],
                }
            ],
            "spillover_systems": [],
            "flows": [],
        }

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            lambda *args, **kwargs: fake_results,
        )
        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.extract_web_map_signals",
            lambda *args, **kwargs: fake_signals,
        )
        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.format_web_map_signals_context",
            lambda signals: "## STRUCTURED WEB MAP SIGNALS\n\n- Receiving: CHN",
        )

        client = MockClient()
        advisor = MetacouplingAssistant(
            llm_client=client,
            web_search=True,
            web_structured_extraction=True,
        )

        result = advisor.analyze("Brazil soybean exports")

        assert result.web_map_signals == fake_signals
        assert "STRUCTURED WEB MAP SIGNALS" in client.last_messages[0].content

    def test_web_sources_footer_uses_w_prefix(self):
        from metacouplingllm.core import MetacouplingAssistant

        results = [
            {"title": "Page A", "url": "https://a.com", "snippet": "snip"},
            {"title": "Page B", "url": "https://b.com", "snippet": "snip"},
        ]
        footer = MetacouplingAssistant._format_web_sources(results)
        assert "[W1]" in footer
        assert "[W2]" in footer
        assert "WEB SOURCES" in footer

    def test_anthropic_adapter_auto_wires_anthropic_web_backend(
        self, monkeypatch
    ):
        """AnthropicAdapter → AnthropicWebSearchBackend gets auto-wired."""
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.knowledge.websearch import (
            AnthropicWebSearchBackend,
        )
        from metacouplingllm.llm.client import AnthropicAdapter, LLMResponse

        # Subclass stubs out .chat so the real SDK call never fires — the
        # isinstance() branch in _run_web_search only needs the class
        # hierarchy intact, not a functional Anthropic client.
        class StubAnthropicAdapter(AnthropicAdapter):
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        adapter = StubAnthropicAdapter(client=None, model="claude-opus-4-7")

        captured: dict[str, object] = {}

        def fake_search_web(*args, **kwargs):
            captured["backend"] = kwargs.get("backend")
            return []

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            fake_search_web,
        )

        advisor = MetacouplingAssistant(
            llm_client=adapter,
            web_search=True,
        )
        advisor.analyze("Brazil soybean exports")

        backend = captured["backend"]
        assert isinstance(backend, AnthropicWebSearchBackend)
        # The adapter's model should flow through to the backend
        assert backend.model == "claude-opus-4-7"

    def test_openai_adapter_still_auto_wires_openai_backend(
        self, monkeypatch
    ):
        """Regression: OpenAIAdapter wiring is unchanged after the Anthropic
        branch was added."""
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.knowledge.websearch import OpenAIWebSearchBackend
        from metacouplingllm.llm.client import LLMResponse, OpenAIAdapter

        class StubOpenAIAdapter(OpenAIAdapter):
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        adapter = StubOpenAIAdapter(client=None, model="gpt-5")

        captured: dict[str, object] = {}

        def fake_search_web(*args, **kwargs):
            captured["backend"] = kwargs.get("backend")
            return []

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            fake_search_web,
        )

        advisor = MetacouplingAssistant(
            llm_client=adapter,
            web_search=True,
        )
        advisor.analyze("Brazil soybean exports")

        backend = captured["backend"]
        assert isinstance(backend, OpenAIWebSearchBackend)
        assert backend.model == "gpt-5"

    def test_generic_client_gets_no_backend(self, monkeypatch):
        """Non-OpenAI, non-Anthropic clients fall through to DuckDuckGo."""
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import LLMResponse

        class GenericClient:
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        captured: dict[str, object] = {}

        def fake_search_web(*args, **kwargs):
            captured["backend"] = kwargs.get("backend")
            return []

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            fake_search_web,
        )

        advisor = MetacouplingAssistant(
            llm_client=GenericClient(),
            web_search=True,
        )
        advisor.analyze("Brazil soybean exports")

        assert captured["backend"] is None

    def test_advisor_prints_web_search_via_anthropic_label(
        self, monkeypatch, capsys,
    ):
        """On a successful Claude web_search run, the advisor prints
        'Web search via Claude web_search (...)' with no fallback line."""
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import AnthropicAdapter, LLMResponse

        class StubAnthropicAdapter(AnthropicAdapter):
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        adapter = StubAnthropicAdapter(client=None, model="claude-opus-4-7")

        def fake_search_web(*args, **kwargs):
            meta = kwargs.get("metadata")
            if meta is not None:
                meta["backend_used"] = "Claude web_search (claude-opus-4-7)"
            return [{"title": "X", "snippet": "s", "url": "https://x.com"}]

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            fake_search_web,
        )

        advisor = MetacouplingAssistant(llm_client=adapter, web_search=True)
        advisor.analyze("Brazil soybean exports")

        out = capsys.readouterr().out
        assert (
            "[MetacouplingAssistant] Web search via Claude web_search "
            "(claude-opus-4-7)"
        ) in out
        assert "fallback after" not in out

    def test_advisor_prints_web_search_via_duckduckgo_fallback_label(
        self, monkeypatch, capsys,
    ):
        """When the configured backend fails, the advisor prints the
        DuckDuckGo line with an explicit fallback reason."""
        from metacouplingllm.core import MetacouplingAssistant
        from metacouplingllm.llm.client import AnthropicAdapter, LLMResponse

        class StubAnthropicAdapter(AnthropicAdapter):
            def chat(self, messages, temperature=0.7, max_tokens=None):
                return LLMResponse(content="Test response.")

        adapter = StubAnthropicAdapter(client=None, model="claude-opus-4-7")

        def fake_search_web(*args, **kwargs):
            meta = kwargs.get("metadata")
            if meta is not None:
                meta["backend_used"] = "DuckDuckGo (fan-out + top-up)"
                meta["fallback_from"] = (
                    "Claude web_search (claude-opus-4-7) raised: API down"
                )
            return [{"title": "X", "snippet": "s", "url": "https://x.com"}]

        monkeypatch.setattr(
            "metacouplingllm.knowledge.websearch.search_web",
            fake_search_web,
        )

        advisor = MetacouplingAssistant(llm_client=adapter, web_search=True)
        advisor.analyze("Brazil soybean exports")

        out = capsys.readouterr().out
        assert (
            "[MetacouplingAssistant] Web search via DuckDuckGo "
            "(fan-out + top-up) \u2014 fallback after Claude web_search "
            "(claude-opus-4-7) raised: API down"
        ) in out


# ---------------------------------------------------------------------------
# JSON extraction (including truncated JSON repair)
# ---------------------------------------------------------------------------

class TestExtractJsonObject:
    def test_bare_json(self):
        result = _extract_json_object('{"key": "value"}')
        assert result == {"key": "value"}

    def test_fenced_json(self):
        text = 'Here is the data:\n```json\n{"focal_country": "USA"}\n```'
        result = _extract_json_object(text)
        assert result == {"focal_country": "USA"}

    def test_json_with_surrounding_text(self):
        text = 'Based on the analysis:\n{"status": "ok"}\nThat is all.'
        result = _extract_json_object(text)
        assert result == {"status": "ok"}

    def test_empty_input(self):
        assert _extract_json_object("") is None
        assert _extract_json_object("   ") is None
        assert _extract_json_object(None) is None  # type: ignore

    def test_no_json(self):
        assert _extract_json_object("Hello, world!") is None

    def test_truncated_json_repair(self):
        """Truncated JSON (cut by max_tokens) should be repaired."""
        truncated = '{"focal_country": "USA", "receiving_countries": ["CHN", "MEX"'
        result = _extract_json_object(truncated)
        assert result is not None
        assert result["focal_country"] == "USA"
        assert "CHN" in result["receiving_countries"]

    def test_truncated_json_with_trailing_key(self):
        """Truncated mid-key should still repair."""
        truncated = '{"focal_country": "BRA", "receiving_countries": ["CHN"], "flows":'
        result = _extract_json_object(truncated)
        # Should repair to at least get focal_country
        assert result is not None
        assert result["focal_country"] == "BRA"

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}, "list": [1, 2, 3]}'
        result = _extract_json_object(text)
        assert result == {"outer": {"inner": 42}, "list": [1, 2, 3]}

    def test_returns_none_for_json_array(self):
        """JSON arrays should not be returned (we expect objects)."""
        assert _extract_json_object('[1, 2, 3]') is None

    def test_complex_map_data(self):
        """Realistic map data extraction response."""
        text = '''```json
{
    "focal_country": "United States",
    "receiving_countries": ["China", "Mexico", "Japan"],
    "spillover_countries": ["Brazil", "Argentina"],
    "flows": [
        {
            "category": "matter",
            "direction": "United States → China",
            "description": "Corn exports"
        }
    ]
}
```'''
        result = _extract_json_object(text)
        assert result is not None
        assert result["focal_country"] == "United States"
        assert len(result["receiving_countries"]) == 3
        assert len(result["flows"]) == 1
