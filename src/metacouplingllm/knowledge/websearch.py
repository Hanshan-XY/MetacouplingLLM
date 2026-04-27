"""
Web search integration for grounding metacoupling analyses.

Provides a lightweight wrapper around DuckDuckGo search (free, no API
key) to inject real-world context into the LLM prompt before analysis.
This is especially useful for niche topics where the LLM's pre-training
knowledge may be limited (e.g., "Michigan's pork exports").

Three backends are tried in order:

1. ``ddgs`` (pip install ddgs) — fast, full-featured
2. ``duckduckgo_search`` — older package, same author
3. **stdlib fallback** — ``urllib`` + ``html.parser``, zero external
   dependencies. Works on Google Colab and other restricted
   environments where the Rust-compiled ``primp`` dependency cannot be
   installed.
"""

from __future__ import annotations

import html.parser
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

_QUERY_FILLER = frozenset(
    (
        "a an analyze analysed analyzed analyzes analyzing analysis and apply "
        "applying assess assessed assessing assesses case consider considering "
        "examines examining explain explains focus focused focusing for from "
        "how impact impacts in investigate investigating looking my of on or "
        "our paper project question questions research researcher researchers "
        "review reviews study studying the to topic topics what will with"
    ).split()
)

_STRUCTURAL_TERMS = frozenset(
    (
        "chain chains commodity commodities coupling couplings effect effects "
        "environment environmental export exported exporter exporters exports "
        "flow flows food framework frameworks import imported importer "
        "importers imports industry industries market markets metacoupling "
        "partner partners production productions sector sectors supply system "
        "systems telecoupling trade traded trader traders trading transfer "
        "transfers"
    ).split()
)

_TRADE_HINTS = frozenset(
    (
        "destination destinations export exported exporter exporters exports "
        "import imported importer importers imports market markets partner "
        "partners trade traded trader traders trading"
    ).split()
)


class WebSearchBackend(Protocol):
    """Protocol for pluggable web-search backends."""

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        """Return search results as ``title``/``snippet``/``url`` dicts."""
        ...


@dataclass
class OpenAIWebSearchBackend:
    """Web-search backend powered by OpenAI Responses API ``web_search``.

    Results are normalized to the same ``title``/``snippet``/``url`` shape
    used by the existing DuckDuckGo backends so downstream consumers do not
    need to change.
    """

    client: Any
    model: str = "gpt-5"
    reasoning: str = "default"
    external_web_access: bool = True
    allowed_domains: list[str] | None = None
    user_location: dict[str, object] | None = None

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        tool: dict[str, object] = {
            "type": "web_search",
            "external_web_access": self.external_web_access,
        }
        if self.allowed_domains:
            tool["filters"] = {"allowed_domains": self.allowed_domains}
        if self.user_location:
            tool["user_location"] = self.user_location

        prompt = (
            f"Search the web for: {query.strip()}\n"
            f"Return up to {max_results} highly relevant results as JSON only.\n"
            "Schema: "
            '{"results":[{"title":"...","url":"...","snippet":"..."}]}\n'
            "Rules:\n"
            "- Use only information grounded in the web search results.\n"
            "- Keep snippets concise and factual.\n"
            "- Do not invent URLs.\n"
            "- Return JSON only.\n"
        )

        kwargs: dict[str, object] = {
            "model": self.model,
            "tools": [tool],
            "tool_choice": "auto",
            "include": ["web_search_call.action.sources"],
            "input": prompt,
        }
        if self.reasoning and self.reasoning != "default":
            kwargs["reasoning"] = {"effort": self.reasoning}

        response = self.client.responses.create(**kwargs)
        output_text = getattr(response, "output_text", "") or ""
        raw_obj = _extract_json_object(output_text)
        parsed_results = []
        if isinstance(raw_obj, dict):
            parsed_results = _normalise_backend_results(
                raw_obj.get("results", []),
                max_results=max_results,
            )

        # Fallback: if the model didn't emit JSON results cleanly, recover
        # from the included source list so the search still yields usable URLs.
        source_results = _extract_openai_source_results(response, max_results)
        if parsed_results:
            merged = []
            seen: set[tuple[str, str]] = set()
            _merge_unique_results(
                merged,
                parsed_results,
                seen=seen,
                max_results=max_results,
            )
            _merge_unique_results(
                merged,
                source_results,
                seen=seen,
                max_results=max_results,
            )
            return merged[:max_results]
        return source_results[:max_results]


# Model → web_search tool version compatibility. Used by
# AnthropicWebSearchBackend when ``tool_version`` is left as None.
# Per Anthropic docs, web_search_20260209 (with dynamic filtering) is
# supported on Opus 4.7, Opus 4.6, and Sonnet 4.6. Older Claude 4 models
# must use web_search_20250305 (no dynamic filtering).
_WEB_SEARCH_MODEL_VERSIONS: dict[str, str] = {
    # Supports web_search_20260209 (dynamic filtering)
    "claude-opus-4-7": "web_search_20260209",
    "claude-opus-4-6": "web_search_20260209",
    "claude-sonnet-4-6": "web_search_20260209",
    # Older Claude 4 models — use web_search_20250305 (no dynamic filtering)
    "claude-opus-4-5": "web_search_20250305",
    "claude-opus-4-1": "web_search_20250305",
    "claude-opus-4-0": "web_search_20250305",
    "claude-sonnet-4-5": "web_search_20250305",
    "claude-sonnet-4-0": "web_search_20250305",
    "claude-haiku-4-5": "web_search_20250305",  # conservative default
}


def _infer_web_search_tool_version(model: str) -> str:
    """Pick the ``web_search`` tool version best supported by *model*.

    Returns ``web_search_20260209`` (dynamic filtering) for Opus 4.7, Opus
    4.6, and Sonnet 4.6 — the models Anthropic explicitly lists as
    supporting the newer version. Returns ``web_search_20250305`` (no
    dynamic filtering) for older Claude 4 models. Unknown models default
    to the newer version; the caller can pass ``tool_version`` explicitly
    to override.
    """
    known = _WEB_SEARCH_MODEL_VERSIONS.get(model)
    if known:
        return known
    return "web_search_20260209"


@dataclass
class AnthropicWebSearchBackend:
    """Web-search backend powered by the Anthropic Messages API ``web_search``.

    By default the tool version is **auto-selected from ``model``** via
    :func:`_infer_web_search_tool_version`:

    ==========================  ==========================
    Model                       tool_version
    ==========================  ==========================
    claude-opus-4-7 (default)   web_search_20260209
    claude-opus-4-6             web_search_20260209
    claude-sonnet-4-6           web_search_20260209
    claude-opus-4-5 / 4-1 / 4-0 web_search_20250305
    claude-sonnet-4-5 / 4-0     web_search_20250305
    claude-haiku-4-5            web_search_20250305
    unknown                     web_search_20260209
    ==========================  ==========================

    Pass an explicit ``tool_version`` to override (useful when a new model
    ships before the lookup table is updated).

    Snippets are pulled from citations on Claude's text blocks (each citation
    carries ``url``, ``title``, and up to 150 chars of ``cited_text``). If
    Claude returns no citations, the backend falls back to the raw
    ``web_search_tool_result`` blocks for ``url`` + ``title`` + ``page_age``
    (the underlying ``encrypted_content`` is opaque to callers by design).

    Prompt caching is intentionally not used: a web-search call's cacheable
    prefix (tool definition plus short user prompt) is well under Opus
    4.7's 4096-token minimum, so caching would silently no-op.
    """

    client: Any
    model: str = "claude-opus-4-7"
    tool_version: str | None = None   # None = auto-select from model
    max_uses: int = 5
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: dict[str, object] | None = None
    max_tokens: int = 8192

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        tool_version = (
            self.tool_version
            or _infer_web_search_tool_version(self.model)
        )
        tool: dict[str, object] = {
            "type": tool_version,
            "name": "web_search",
            "max_uses": self.max_uses,
        }
        if self.allowed_domains:
            tool["allowed_domains"] = self.allowed_domains
        if self.blocked_domains:
            tool["blocked_domains"] = self.blocked_domains
        if self.user_location:
            tool["user_location"] = self.user_location

        prompt = (
            f"Search the web for: {query.strip()}\n"
            f"Return up to {max_results} highly relevant sources. "
            "Cite every factual statement you make inline so that every "
            "source you used appears in a citation."
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[tool],
            messages=[{"role": "user", "content": prompt}],
        )

        return _extract_anthropic_web_results(response, max_results)


@dataclass
class GeminiWebSearchBackend:
    """Web-search backend powered by Google Gemini's grounding tool.

    Built against the new ``google.genai`` SDK. Calls
    ``client.models.generate_content`` with the ``google_search`` tool
    enabled (Gemini 2.x grounding). The model performs the search and
    synthesises results; we parse the response for a JSON list of
    ``title``/``snippet``/``url`` items, falling back to
    ``grounding_metadata.grounding_chunks`` for URL+title recovery if
    JSON parsing fails.

    Parameters
    ----------
    client:
        A ``google.genai.Client`` instance.
    model:
        Gemini model identifier supporting grounding (e.g.
        ``"gemini-2.5-flash"`` or ``"gemini-2.5-pro"``).
    max_tokens:
        Output budget for the search-and-summarise call.
    """

    client: Any
    model: str = "gemini-2.5-flash"
    max_tokens: int = 8192

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        prompt = (
            f"Search the web for: {query.strip()}\n"
            f"Return up to {max_results} highly relevant results as JSON only.\n"
            "Schema: "
            '{"results":[{"title":"...","url":"...","snippet":"..."}]}\n'
            "Rules:\n"
            "- Use only information grounded in the web search results.\n"
            "- Keep snippets concise and factual.\n"
            "- Do not invent URLs.\n"
            "- Return JSON only.\n"
        )

        try:
            # google.genai 1.x: tools accept a list of dicts; the
            # google_search tool is enabled with an empty config.
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "max_output_tokens": self.max_tokens,
                    "tools": [{"google_search": {}}],
                },
            )
        except Exception:
            return []

        # Pull the model's text (may raise if blocked).
        try:
            output_text = response.text or ""
        except Exception:
            output_text = ""

        raw_obj = _extract_json_object(output_text)
        parsed_results: list[dict[str, str]] = []
        if isinstance(raw_obj, dict):
            parsed_results = _normalise_backend_results(
                raw_obj.get("results", []),
                max_results=max_results,
            )

        # Fallback: walk grounding_metadata.grounding_chunks for url + title.
        grounding_results = _extract_gemini_grounding_results(
            response, max_results
        )
        if parsed_results:
            merged: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            _merge_unique_results(
                merged,
                parsed_results,
                seen=seen,
                max_results=max_results,
            )
            _merge_unique_results(
                merged,
                grounding_results,
                seen=seen,
                max_results=max_results,
            )
            return merged[:max_results]
        return grounding_results[:max_results]


@dataclass
class GrokWebSearchBackend:
    """Web-search backend powered by xAI Grok's Live Search tool.

    Live Search is invoked by passing ``extra_body={"search_parameters":
    {...}}`` to a chat-completions call. The response includes
    ``citations`` (a list of URLs) plus the assistant message with
    inline citations. We ask the model to return JSON; if that fails,
    we fall back to ``response.citations`` for URL recovery.

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` instance configured with
        ``base_url="https://api.x.ai/v1"``.
    model:
        Grok model identifier (e.g., ``"grok-3"``, ``"grok-2"``).
    search_mode:
        xAI's ``mode`` for Live Search: ``"auto"`` (default — model
        decides), ``"on"`` (always search), or ``"off"`` (no search).
    max_search_results:
        Upper bound on internal search calls per request (xAI default
        is 15).
    sources:
        Override the default Live Search sources. Default is
        ``[{"type": "web"}, {"type": "x"}]`` — web + Twitter/X.
    max_tokens:
        Output budget for the search-and-summarise call.
    """

    client: Any
    model: str = "grok-3"
    search_mode: str = "auto"
    max_search_results: int = 15
    sources: list[dict[str, object]] | None = None
    max_tokens: int = 8192

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        prompt = (
            f"Search the web for: {query.strip()}\n"
            f"Return up to {max_results} highly relevant results as JSON only.\n"
            "Schema: "
            '{"results":[{"title":"...","url":"...","snippet":"..."}]}\n'
            "Rules:\n"
            "- Use only information grounded in your live search results.\n"
            "- Keep snippets concise and factual.\n"
            "- Do not invent URLs.\n"
            "- Return JSON only.\n"
        )

        sources = self.sources or [{"type": "web"}, {"type": "x"}]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                extra_body={
                    "search_parameters": {
                        "mode": self.search_mode,
                        "max_search_results": self.max_search_results,
                        "sources": sources,
                    },
                },
            )
        except Exception:
            return []

        message_text = ""
        if getattr(response, "choices", None):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                message_text = getattr(message, "content", "") or ""

        raw_obj = _extract_json_object(message_text)
        parsed_results: list[dict[str, str]] = []
        if isinstance(raw_obj, dict):
            parsed_results = _normalise_backend_results(
                raw_obj.get("results", []),
                max_results=max_results,
            )

        # Fallback: harvest citations (URLs) when JSON parsing fails.
        citation_results = _extract_grok_citation_results(
            response, max_results
        )
        if parsed_results:
            merged: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            _merge_unique_results(
                merged,
                parsed_results,
                seen=seen,
                max_results=max_results,
            )
            _merge_unique_results(
                merged,
                citation_results,
                seen=seen,
                max_results=max_results,
            )
            return merged[:max_results]
        return citation_results[:max_results]


def _extract_gemini_grounding_results(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Walk a Gemini response for ``grounding_chunks[].web`` URL+title pairs."""
    candidates = getattr(response, "candidates", None) or []
    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for candidate in candidates:
        meta = getattr(candidate, "grounding_metadata", None)
        if meta is None:
            continue
        chunks = getattr(meta, "grounding_chunks", None) or []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if web is None:
                continue
            url = (getattr(web, "uri", "") or "").strip()
            title = (getattr(web, "title", "") or "").strip() or url
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append({"title": title, "snippet": "", "url": url})
            if len(out) >= max_results:
                return out
    return out


def _extract_grok_citation_results(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Harvest ``response.citations`` (URLs) as fallback web results."""
    citations = getattr(response, "citations", None) or []
    out: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for cite in citations:
        # `citations` may be a list of strings (URLs) or dict-like objects
        # depending on xAI SDK version. Handle both shapes.
        if isinstance(cite, str):
            url = cite.strip()
            title = url
            snippet = ""
        elif isinstance(cite, dict):
            url = str(cite.get("url") or "").strip()
            title = str(cite.get("title") or url).strip()
            snippet = str(cite.get("snippet") or cite.get("excerpt") or "").strip()
        else:
            url = str(getattr(cite, "url", "") or "").strip()
            title = str(getattr(cite, "title", url) or url).strip()
            snippet = str(getattr(cite, "snippet", "") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append({"title": title, "snippet": snippet, "url": url})
        if len(out) >= max_results:
            break
    return out


# -------------------------------------------------------------------
# Stdlib fallback — zero external dependencies
# -------------------------------------------------------------------

class _DuckDuckGoLiteParser(html.parser.HTMLParser):
    """Minimal parser for DuckDuckGo Lite HTML results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._current: dict[str, str] = {}
        self._text_buf: list[str] = []

    # DuckDuckGo Lite wraps each result link in
    # <a rel="nofollow" class="result-link" href="…">
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("class") == "result-link":
            self._in_link = True
            self._current = {"url": attr.get("href", ""), "title": "", "snippet": ""}
            self._text_buf = []
        elif tag == "td" and attr.get("class") == "result-snippet":
            self._in_snippet = True
            self._text_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            self._current["title"] = " ".join(self._text_buf).strip()
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            self._current["snippet"] = " ".join(self._text_buf).strip()
            if self._current.get("url"):
                self.results.append(self._current)
            self._current = {}

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_snippet:
            self._text_buf.append(data)


def _resolve_ddg_url(raw: str) -> str:
    """Extract the real destination URL from a DuckDuckGo redirect link.

    DuckDuckGo Lite wraps result links as
    ``//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com&rut=…``.
    This function extracts the ``uddg`` parameter value, falling back
    to the original URL if it doesn't match the redirect pattern.
    """
    if "duckduckgo.com/l/" in raw and "uddg=" in raw:
        parsed = urllib.parse.urlparse(raw if "://" in raw else "https:" + raw)
        params = urllib.parse.parse_qs(parsed.query)
        uddg = params.get("uddg", [None])[0]
        if uddg:
            return uddg
    return raw


def _search_stdlib(query: str, max_results: int) -> list[dict[str, str]]:
    """Search DuckDuckGo using only the standard library.

    Handles network errors, timeouts, and malformed HTML gracefully
    by returning an empty list on failure.
    """
    # Guard against excessively long queries that could exceed URL limits
    if len(query) > 2000:
        query = query[:2000]

    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; metacoupling-python/0.1; "
                "+https://github.com/metacoupling/metacoupling)"
            ),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout,
            OSError) as exc:
        print(f"[WebSearch] stdlib network error: {exc}")
        return []

    try:
        parser = _DuckDuckGoLiteParser()
        parser.feed(body)
    except Exception as exc:
        print(f"[WebSearch] HTML parsing error: {exc}")
        return []

    # Resolve redirect URLs to actual destinations
    for r in parser.results:
        r["url"] = _resolve_ddg_url(r["url"])

    return parser.results[:max_results]


def _search_ddgs(query: str, max_results: int) -> list[dict[str, str]]:
    """Search using the modern ``ddgs`` backend."""
    from ddgs import DDGS  # type: ignore[import-untyped]

    results: list[dict[str, str]] = []
    for r in DDGS().text(query, max_results=max_results):
        results.append({
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": _resolve_ddg_url(r.get("href", "")),
        })
    return results


def _search_duckduckgo_search(
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    """Search using the legacy ``duckduckgo_search`` backend."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"duckduckgo_search(\..*)?",
        )
        from duckduckgo_search import DDGS as DDGS2  # type: ignore[import-untyped]

        results: list[dict[str, str]] = []
        for r in DDGS2().text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": _resolve_ddg_url(r.get("href", "")),
            })
    return results


def _result_key(result: dict[str, str]) -> tuple[str, str]:
    """Build a stable dedupe key for a search result."""
    url = result.get("url", "").strip().lower().rstrip("/")
    if url:
        return ("url", url)
    title = " ".join(result.get("title", "").lower().split())
    snippet = " ".join(result.get("snippet", "").lower().split())
    return ("text", f"{title}|{snippet}")


def _normalise_backend_results(
    results: object,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    """Normalise backend output to ``title``/``snippet``/``url`` dicts."""
    if not isinstance(results, list):
        return []

    normalised: list[dict[str, str]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        normalised.append({
            "title": title,
            "snippet": snippet,
            "url": url,
        })
        if len(normalised) >= max_results:
            break
    return normalised


def _extract_openai_source_results(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Best-effort extraction of ``url``/``title`` source rows from a response."""
    raw: object
    if hasattr(response, "model_dump"):
        try:
            raw = response.model_dump()
        except Exception:
            raw = None
    elif isinstance(response, dict):
        raw = response
    else:
        raw = None

    if not isinstance(raw, (dict, list)):
        return []

    seen_urls: set[str] = set()
    found: list[dict[str, str]] = []

    def _walk(node: object) -> None:
        if len(found) >= max_results:
            return
        if isinstance(node, dict):
            url = node.get("url")
            title = node.get("title")
            if isinstance(url, str) and url.strip():
                clean_url = url.strip()
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    found.append({
                        "title": str(title).strip() if title else clean_url,
                        "snippet": str(
                            node.get("snippet")
                            or node.get("excerpt")
                            or node.get("description")
                            or ""
                        ).strip(),
                        "url": clean_url,
                    })
                    if len(found) >= max_results:
                        return
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(raw)
    return found[:max_results]


def _extract_anthropic_web_results(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Extract ``title``/``snippet``/``url`` rows from an Anthropic response.

    Primary path: walk the ``citations`` lists on each ``text`` content block
    — every citation carries the source URL, title, and up to 150 chars of
    ``cited_text`` (the snippet). Dedupe by URL, take the first citation's
    ``cited_text`` as the snippet when a URL is cited multiple times.

    Fallback path: when Claude returned search results but no citations,
    walk the ``web_search_tool_result`` blocks for ``url`` + ``title``.
    ``encrypted_content`` is opaque by design, so snippets in this path
    degrade to just the ``page_age`` timestamp.
    """
    raw: object
    if hasattr(response, "model_dump"):
        try:
            raw = response.model_dump()
        except Exception:
            raw = None
    elif isinstance(response, dict):
        raw = response
    else:
        raw = None

    if not isinstance(raw, dict):
        return []

    content = raw.get("content")
    if not isinstance(content, list):
        return []

    seen_urls: set[str] = set()
    found: list[dict[str, str]] = []

    # Primary: citations on text blocks
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        citations = block.get("citations")
        if not isinstance(citations, list):
            continue
        for cit in citations:
            if not isinstance(cit, dict):
                continue
            if cit.get("type") != "web_search_result_location":
                continue
            url = str(cit.get("url", "") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = str(cit.get("title", "") or "").strip() or url
            snippet = str(cit.get("cited_text", "") or "").strip()
            found.append({"title": title, "snippet": snippet, "url": url})
            if len(found) >= max_results:
                return found

    # Fallback: web_search_tool_result blocks when no citations were emitted
    if not found:
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "web_search_tool_result":
                continue
            tr_content = block.get("content")
            if not isinstance(tr_content, list):
                continue
            for res in tr_content:
                if not isinstance(res, dict):
                    continue
                if res.get("type") != "web_search_result":
                    continue
                url = str(res.get("url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = str(res.get("title", "") or "").strip() or url
                page_age = str(res.get("page_age", "") or "").strip()
                snippet = f"(page age: {page_age})" if page_age else ""
                found.append({"title": title, "snippet": snippet, "url": url})
                if len(found) >= max_results:
                    return found

    return found[:max_results]


def _merge_unique_results(
    merged: list[dict[str, str]],
    new_results: list[dict[str, str]],
    *,
    seen: set[tuple[str, str]],
    max_results: int,
) -> None:
    """Append unique results to *merged* until *max_results* is reached."""
    for result in new_results:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
        if len(merged) >= max_results:
            return


def _extract_query_phrases(query: str, max_words: int = 4) -> list[str]:
    """Return overlapping word phrases from longest to shortest."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", query)
    phrases: list[str] = []
    seen: set[str] = set()
    for size in range(max_words, 0, -1):
        for idx in range(0, len(words) - size + 1):
            phrase = " ".join(words[idx: idx + size]).strip()
            if not phrase:
                continue
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            phrases.append(phrase)
    return phrases


def _strip_possessive(text: str) -> str:
    """Strip a trailing possessive marker from *text*."""
    return re.sub(r"(?:'s|’s)$", "", text.strip(), flags=re.IGNORECASE)


def _detect_subnational_focus(query: str) -> tuple[str | None, str | None]:
    """Return ADM1 place and parent country when the query is subnational."""
    try:
        from metacouplingllm.knowledge.adm1_pericoupling import (
            get_adm1_info,
            resolve_adm1_code,
        )
        from metacouplingllm.knowledge.countries import (
            get_country_name,
            resolve_country_code,
        )
    except Exception:
        return None, None

    for phrase in _extract_query_phrases(query):
        cleaned = _strip_possessive(phrase)
        if not cleaned:
            continue
        code = resolve_adm1_code(cleaned)
        if not code:
            continue
        info = get_adm1_info(code)
        if not info:
            return cleaned, None
        country_name = info.get("country_name")
        country_code = resolve_country_code(country_name or "")
        if country_code:
            country_name = get_country_name(country_code)
        return info.get("name", cleaned), country_name

    return None, None


def _extract_topic_terms(
    query: str,
    *,
    focal_place: str | None,
    focal_country: str | None,
    max_terms: int = 3,
) -> list[str]:
    """Extract likely commodity/topic terms from a research query."""
    lowered = query.replace("’", "'").lower()

    # Strip focal place/country names AND their ISO codes so they
    # don't appear as topic terms (e.g., "usa" from "USA").
    try:
        from metacouplingllm.knowledge.countries import resolve_country_code

        _extra_strips: list[str] = []
        for phrase in (focal_place, focal_country):
            if phrase:
                code = resolve_country_code(phrase)
                if code:
                    _extra_strips.append(code.lower())
    except Exception:
        _extra_strips = []

    for phrase in (focal_place, focal_country):
        if not phrase:
            continue
        lowered = re.sub(
            rf"\b{re.escape(phrase.lower())}(?:'s)?\b",
            " ",
            lowered,
        )
    for code_lower in _extra_strips:
        lowered = re.sub(rf"\b{re.escape(code_lower)}\b", " ", lowered)

    topic_terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z][a-z'-]{2,}", lowered):
        token = _strip_possessive(token.lower())
        if (
            token in _QUERY_FILLER
            or token in _STRUCTURAL_TERMS
            or token in _TRADE_HINTS
        ):
            continue
        if token in seen:
            continue
        seen.add(token)
        topic_terms.append(token)
        if len(topic_terms) >= max_terms:
            break

    return topic_terms


def _has_trade_intent(query: str) -> bool:
    """Return True when the research query is about trade/export partners."""
    tokens = {
        _strip_possessive(token.lower())
        for token in re.findall(r"[A-Za-z][A-Za-z’’-]*", query)
    }
    return any(token in _TRADE_HINTS for token in tokens)


def _detect_national_focus(query: str) -> str | None:
    """Extract a country name from the query when no subnational region found.

    E.g., ``"corn production and exports in USA"`` → ``"United States"``.
    """
    try:
        from metacouplingllm.knowledge.countries import (
            get_country_name,
            resolve_country_code,
        )
    except Exception:
        return None

    for phrase in _extract_query_phrases(query):
        cleaned = _strip_possessive(phrase)
        if not cleaned:
            continue
        code = resolve_country_code(cleaned)
        if code:
            return get_country_name(code) or cleaned

    return None


def _build_search_queries(query: str) -> list[str]:
    """Expand a raw research query into higher-signal search variants."""
    base_query = " ".join(query.split())
    if not base_query:
        return []

    if not _has_trade_intent(base_query):
        return [base_query]

    # Try subnational focus first (e.g., "Michigan pork exports")
    focal_place, focal_country = _detect_subnational_focus(base_query)

    # If no subnational focus, try national (e.g., "USA corn exports")
    if not focal_place:
        focal_country = _detect_national_focus(base_query)
        if not focal_country:
            return [base_query]
        focal_place = focal_country

    topic_terms = _extract_topic_terms(
        base_query,
        focal_place=focal_place,
        focal_country=focal_country,
    )

    queries: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        cleaned = " ".join(candidate.split()).strip()
        if not cleaned:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(cleaned)

    if topic_terms:
        for term in topic_terms:
            add(f"{focal_place} {term} export destinations")
            add(f"{focal_place} {term} trade partners")
            add(f"{focal_place} {term} export markets")
            if focal_country:
                add(f"{focal_country} {term} exports top destinations")
                add(f"{focal_country} {term} export markets")
    else:
        add(f"{focal_place} export destinations")
        add(f"{focal_place} trade partners")
        add(f"{focal_place} export markets")
        if focal_country:
            add(f"{focal_country} exports top destinations")
            add(f"{focal_country} export markets")

    add(base_query)
    return queries


def _search_single_query(
    query: str,
    max_results: int,
) -> list[dict[str, str]]:
    """Search one query string across all backends and merge results."""
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    backends = (
        ("ddgs", _search_ddgs),
        ("duckduckgo_search", _search_duckduckgo_search),
        ("stdlib fallback", _search_stdlib),
    )

    for backend_name, backend in backends:
        try:
            results = backend(query, max_results)
        except ImportError:
            continue
        except Exception as exc:
            print(f"[WebSearch] {backend_name} backend failed: {exc}")
            continue

        if not results:
            continue

        _merge_unique_results(
            merged, results, seen=seen, max_results=max_results,
        )
        if len(merged) >= max_results:
            break

    return merged[:max_results]


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

_DUCKDUCKGO_LABEL = "DuckDuckGo (fan-out + top-up)"


def _backend_display_name(backend: object | None) -> str:
    """Return a human label describing *backend* for logs/console output."""
    if backend is None:
        return _DUCKDUCKGO_LABEL
    if isinstance(backend, AnthropicWebSearchBackend):
        return f"Claude web_search ({backend.model})"
    if isinstance(backend, OpenAIWebSearchBackend):
        return f"OpenAI web_search ({backend.model})"
    return type(backend).__name__


def search_web(
    query: str,
    max_results: int = 5,
    *,
    backend: WebSearchBackend | None = None,
    metadata: dict[str, object] | None = None,
) -> list[dict[str, str]]:
    """Search the web and return top results.

    Parameters
    ----------
    query:
        Search query string.
    max_results:
        Maximum number of results to return.
    backend:
        Optional pluggable backend (e.g., ``OpenAIWebSearchBackend``,
        ``AnthropicWebSearchBackend``). Tried first; falls back to the
        DuckDuckGo cascade on exception or empty result.
    metadata:
        Optional out-dict the caller can pass to learn which backend
        actually produced the returned results. When provided, it is
        populated with:

        - ``"backend_used"``: human label of the backend that produced
          the results (e.g. ``"Claude web_search (claude-opus-4-7)"`` or
          ``"DuckDuckGo (fan-out + top-up)"``).
        - ``"fallback_from"`` *(only present when fallback occurred)*: a
          short reason string like ``"Claude web_search raised: ..."``
          or ``"OpenAI web_search returned 0 results"``.

    Returns
    -------
    A list of dicts, each with ``title``, ``snippet``, and ``url`` keys.
    Returns an empty list if the search fails or the library is not
    installed.

    Notes
    -----
    When *backend* is provided, it is tried first. If it yields no usable
    results or raises, the built-in DuckDuckGo-family backends are used.

    The built-in search path attempts three backends in order, and unique
    results are merged until *max_results* is reached:

    1. ``ddgs`` (pip install ddgs)
    2. ``duckduckgo_search`` (pip install duckduckgo-search)
    3. stdlib-only fallback using ``urllib`` + ``html.parser`` — works
       everywhere including Google Colab without extra packages.
    """
    if max_results <= 0:
        return []

    fallback_reason: str | None = None
    if backend is not None:
        try:
            results = _normalise_backend_results(
                backend.search(query, max_results=max_results),
                max_results=max_results,
            )
        except Exception as exc:
            print(f"[WebSearch] custom backend failed: {exc}")
            fallback_reason = (
                f"{_backend_display_name(backend)} raised: {exc}"
            )
        else:
            if results:
                if metadata is not None:
                    metadata["backend_used"] = _backend_display_name(backend)
                return results[:max_results]
            fallback_reason = (
                f"{_backend_display_name(backend)} returned 0 results"
            )

    queries = _build_search_queries(query)
    if not queries:
        if metadata is not None:
            metadata["backend_used"] = _DUCKDUCKGO_LABEL
            if fallback_reason is not None:
                metadata["fallback_from"] = fallback_reason
        return []

    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    per_query_max = (
        max_results
        if len(queries) == 1
        else max(4, min(max_results, 6))
    )

    for search_query in queries:
        results = _search_single_query(search_query, per_query_max)
        if not results:
            continue

        _merge_unique_results(
            merged,
            results,
            seen=seen,
            max_results=max_results,
        )
        if len(merged) >= max_results:
            break

    # Top-up: when fan-out is used (multiple variants) AND the per-variant
    # cap left us short of max_results, re-issue the raw query at the full
    # budget. The dedup keys ensure variant URLs are not double-counted.
    # This recovers cases where a single query has plenty of unique hits
    # available (e.g., "Michigan pork exports" returns ~47 unique at max=50)
    # that the per-variant cap of ~6 would otherwise throttle away.
    if len(queries) > 1 and len(merged) < max_results:
        topup = _search_single_query(queries[-1], max_results)
        _merge_unique_results(
            merged,
            topup,
            seen=seen,
            max_results=max_results,
        )

    if metadata is not None:
        metadata["backend_used"] = _DUCKDUCKGO_LABEL
        if fallback_reason is not None:
            metadata["fallback_from"] = fallback_reason

    return merged[:max_results]


def format_web_context(results: list[dict[str, str]]) -> str:
    """Format web search results into a prompt-ready context section.

    Parameters
    ----------
    results:
        List of search result dicts from :func:`search_web`.

    Returns
    -------
    A formatted string starting with ``## WEB SEARCH CONTEXT``, or an
    empty string if no results are provided.
    """
    if not results:
        return ""

    lines: list[str] = [
        "## WEB SEARCH CONTEXT",
        "",
        "The following web search snippets provide real-world context "
        "for the research topic. Use this information to ground your "
        "analysis in current facts (e.g., trade volumes, destinations, "
        "policies), but always apply the metacoupling framework "
        "rigorously.",
        "",
        "For subnational trade topics, some snippets may come from "
        "broader country-level searches when place-specific trade-partner "
        "data are sparse. Treat those as proxy context and label them "
        "clearly rather than presenting them as direct subnational facts.",
        "",
        "When you use a fact from a web snippet, cite it inline as "
        "[W1], [W2], etc.  These are distinct from the literature "
        "citations [1], [2], … that may be added separately.",
        "",
    ]

    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        snippet = r.get("snippet", "")
        url = r.get("url", "")
        lines.append(f"[W{i}] **{title}**")
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   Source: {url}")
        lines.append("")

    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, object] | None:
    """Best-effort extraction of a JSON object from model output.

    Handles common LLM output patterns including:
    - Bare JSON
    - JSON inside markdown fenced code blocks
    - JSON surrounded by explanation text
    - Truncated JSON (attempts repair by closing brackets)
    """
    if not text or not text.strip():
        return None

    stripped = text.strip()
    candidates: list[str] = [stripped]

    # Try fenced code blocks first (most reliable)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    candidates.extend(fenced)

    # Try extracting between first { and last }
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(stripped[first:last + 1])

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj

    # Last resort: try to repair truncated JSON by closing open brackets.
    # This handles the case where max_tokens cut the response mid-JSON.
    if first != -1:
        raw = stripped[first:]
        # Count unclosed braces/brackets and close them
        open_braces = raw.count("{") - raw.count("}")
        open_brackets = raw.count("[") - raw.count("]")
        if open_braces > 0 or open_brackets > 0:
            # Strip any trailing incomplete key-value pair
            repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', "", raw)
            repaired = re.sub(r',\s*$', "", repaired)
            repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            try:
                obj = json.loads(repaired)
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                if isinstance(obj, dict):
                    return obj

    return None


def _normalise_web_evidence_ids(
    evidence: object,
    available_ids: set[str],
) -> list[str]:
    """Return valid evidence ids like ``['W1', 'W2']``."""
    if not isinstance(evidence, list):
        return []
    cleaned: list[str] = []
    for item in evidence:
        if not isinstance(item, str):
            continue
        match = re.search(r"W\d+", item.upper())
        if not match:
            continue
        code = match.group(0)
        if code in available_ids and code not in cleaned:
            cleaned.append(code)
    return cleaned


def _coerce_confidence(value: object) -> float:
    """Convert *value* to a bounded confidence score."""
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def extract_web_map_signals(
    query: str,
    results: list[dict[str, str]],
    llm_client: object,
    *,
    min_confidence: float = 0.7,
    max_targets: int = 6,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> dict[str, object] | None:
    """Use an LLM to extract validated map-ready signals from web snippets.

    Returns a dict containing ``focal_country``, ``receiving_systems``,
    ``spillover_systems``, and ``flows`` when extraction succeeds.
    """
    if not results:
        return None

    from metacouplingllm.knowledge.countries import (
        get_country_name,
        resolve_country_code,
    )
    from metacouplingllm.llm.client import Message

    available_ids = {f"W{i}" for i in range(1, len(results) + 1)}
    snippet_lines: list[str] = []
    for idx, result in enumerate(results, 1):
        snippet_lines.append(f"[W{idx}] {result.get('title', '').strip()}")
        snippet_lines.append(f"Snippet: {result.get('snippet', '').strip()}")
        snippet_lines.append(f"URL: {result.get('url', '').strip()}")
        snippet_lines.append("")

    schema = {
        "focal_country": "country name, ISO code, or null",
        "receiving_systems": [
            {
                "country": "country name or ISO code",
                "kind": "direct or proxy",
                "confidence": 0.0,
                "evidence": ["W1"],
                "reason": "short explanation",
            }
        ],
        "spillover_systems": [
            {
                "country": "country name or ISO code",
                "kind": "direct or proxy",
                "confidence": 0.0,
                "evidence": ["W2"],
                "reason": "short explanation",
            }
        ],
        "flows": [
            {
                "category": "matter/capital/information/energy/people/organisms",
                "source_country": "country name or ISO code",
                "target_country": "country name or ISO code",
                "kind": "direct or proxy",
                "confidence": 0.0,
                "evidence": ["W1"],
                "description": "short flow description",
            }
        ],
    }
    system_text = (
        "You extract conservative, map-ready metacoupling signals from web "
        "search snippets. Use only the provided snippets. Do not invent "
        "countries, destinations, or flows. Return JSON only."
    )
    user_text = (
        f"Research query:\n{query.strip()}\n\n"
        "Extract map-ready countries and flows for a metacoupling map.\n"
        "Rules:\n"
        "- Use null or empty lists when uncertain.\n"
        "- Label items as 'proxy' when the snippet is broader than the focal "
        "study but still useful context.\n"
        "- Only include countries that are explicitly supported by the "
        "snippets.\n"
        "- ONLY extract countries and flows that are DIRECTLY RELEVANT to "
        "the research query above. Ignore trade data about unrelated "
        "products, sectors, or commodities, even if they involve the focal "
        "country. For example, if the research is about feed barley, do NOT "
        "extract sheep offal exports or unrelated commodity flows.\n"
        "- receiving_systems = countries that BUY from or RECEIVE "
        "goods/services/capital from the focal country (trade partners, "
        "importers)\n"
        "- spillover_systems = competitors, indirectly affected countries, "
        "or countries that experience environmental/economic spillover "
        "effects\n"
        "- NEVER put competing exporter countries in receiving_systems. "
        "Competitors belong in spillover_systems.\n"
        "- Most flows go FROM the focal country TO receiving countries "
        "(matter, information, energy, people, organisms).\n"
        "- Capital/payment flows go in REVERSE: FROM receiving countries "
        "TO the focal country.\n"
        "- Do NOT create flows from spillover/competitor countries to "
        "anyone. For example, if USA exports corn to Mexico "
        "and Brazil is a competitor, do NOT add a Brazil \u2192 Mexico flow.\n"
        "- Keep at most 6 receiving systems, 6 spillover systems, and 8 flows.\n"
        "- Every item must include evidence ids like W1.\n\n"
        f"JSON schema example:\n{json.dumps(schema, ensure_ascii=True)}\n\n"
        "Web snippets:\n"
        f"{chr(10).join(snippet_lines)}"
    )

    response = llm_client.chat(
        messages=[
            Message(role="system", content=system_text),
            Message(role="user", content=user_text),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_obj = _extract_json_object(response.content)
    if raw_obj is None:
        return None

    def _normalise_country_entry(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        code = resolve_country_code(str(item.get("country", "")).strip())
        if not code:
            return None
        confidence = _coerce_confidence(item.get("confidence", 0.0))
        evidence = _normalise_web_evidence_ids(
            item.get("evidence", []),
            available_ids,
        )
        if confidence < min_confidence or not evidence:
            return None
        kind = str(item.get("kind", "direct")).strip().lower()
        if kind not in {"direct", "proxy"}:
            kind = "direct"
        reason = str(item.get("reason", "")).strip()
        return {
            "country": code,
            "kind": kind,
            "confidence": confidence,
            "evidence": evidence,
            "reason": reason,
        }

    def _normalise_flow_entry(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        src_code = resolve_country_code(str(item.get("source_country", "")).strip())
        tgt_code = resolve_country_code(str(item.get("target_country", "")).strip())
        if not src_code or not tgt_code or src_code == tgt_code:
            return None
        category = str(item.get("category", "")).strip().lower()
        if category not in {
            "matter", "material", "capital", "financial", "information",
            "energy", "people", "organisms",
        }:
            return None
        confidence = _coerce_confidence(item.get("confidence", 0.0))
        evidence = _normalise_web_evidence_ids(
            item.get("evidence", []),
            available_ids,
        )
        if confidence < min_confidence or not evidence:
            return None
        kind = str(item.get("kind", "direct")).strip().lower()
        if kind not in {"direct", "proxy"}:
            kind = "direct"
        description = str(item.get("description", "")).strip()
        src_name = get_country_name(src_code) or src_code
        tgt_name = get_country_name(tgt_code) or tgt_code
        return {
            "category": category,
            "source_country": src_code,
            "target_country": tgt_code,
            "direction": f"{src_name} \u2192 {tgt_name}",
            "description": description,
            "kind": kind,
            "confidence": confidence,
            "evidence": evidence,
        }

    focal_country = resolve_country_code(str(raw_obj.get("focal_country", "")).strip())
    receiving: list[dict[str, object]] = []
    seen_receiving: set[str] = set()
    for item in raw_obj.get("receiving_systems", []) if isinstance(raw_obj.get("receiving_systems", []), list) else []:
        normalized = _normalise_country_entry(item)
        if not normalized:
            continue
        code = str(normalized["country"])
        if code in seen_receiving:
            continue
        seen_receiving.add(code)
        receiving.append(normalized)
        if len(receiving) >= max_targets:
            break

    spillover: list[dict[str, object]] = []
    seen_spillover: set[str] = set()
    for item in raw_obj.get("spillover_systems", []) if isinstance(raw_obj.get("spillover_systems", []), list) else []:
        normalized = _normalise_country_entry(item)
        if not normalized:
            continue
        code = str(normalized["country"])
        if code in seen_spillover:
            continue
        seen_spillover.add(code)
        spillover.append(normalized)
        if len(spillover) >= max_targets:
            break

    flows: list[dict[str, object]] = []
    seen_flows: set[tuple[str, str, str]] = set()
    raw_flows = raw_obj.get("flows", [])
    if isinstance(raw_flows, list):
        for item in raw_flows:
            normalized = _normalise_flow_entry(item)
            if not normalized:
                continue
            key = (
                str(normalized["category"]),
                str(normalized["source_country"]),
                str(normalized["target_country"]),
            )
            if key in seen_flows:
                continue
            seen_flows.add(key)
            flows.append(normalized)
            if len(flows) >= max_targets + 2:
                break

    if not focal_country and receiving and flows:
        focal_country = str(flows[0]["source_country"])

    if not any((focal_country, receiving, spillover, flows)):
        return None

    return {
        "focal_country": focal_country,
        "receiving_systems": receiving,
        "spillover_systems": spillover,
        "flows": flows,
    }


def format_web_map_signals_context(signals: dict[str, object] | None) -> str:
    """Format structured web map signals into a prompt-ready section."""
    if not signals:
        return ""

    lines = [
        "## STRUCTURED WEB MAP SIGNALS",
        "",
        "The following countries and flows were conservatively extracted "
        "from the web snippets and validated for mapping. Treat these as "
        "high-priority map hints, while still reasoning carefully about "
        "direct versus proxy evidence.",
        "",
    ]

    focal = signals.get("focal_country")
    if isinstance(focal, str) and focal:
        lines.append(f"- Focal country: {focal}")

    receiving = signals.get("receiving_systems", [])
    if isinstance(receiving, list) and receiving:
        lines.append("- Receiving systems:")
        for item in receiving:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('country')} ({item.get('kind', 'direct')}, "
                f"confidence={item.get('confidence', 0):.2f}, "
                f"evidence={','.join(item.get('evidence', []))})"
            )

    spillover = signals.get("spillover_systems", [])
    if isinstance(spillover, list) and spillover:
        lines.append("- Spillover systems:")
        for item in spillover:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('country')} ({item.get('kind', 'direct')}, "
                f"confidence={item.get('confidence', 0):.2f}, "
                f"evidence={','.join(item.get('evidence', []))})"
            )

    flows = signals.get("flows", [])
    if isinstance(flows, list) and flows:
        lines.append("- Map-ready flows:")
        for item in flows:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('category')}: {item.get('direction')} "
                f"({item.get('kind', 'direct')}, "
                f"confidence={item.get('confidence', 0):.2f}, "
                f"evidence={','.join(item.get('evidence', []))})"
            )

    return "\n".join(lines)


# -------------------------------------------------------------------
# Post-processing: add inline [WN] citations
# -------------------------------------------------------------------

# Stopwords excluded from keyword matching
_STOP = frozenset(
    "a an the and or but in on of to for is are was were be been by "
    "at as it its with from this that these those not no do does did "
    "has have had will would shall should can could may might must "
    "so if than too very also about more most such each every all "
    "any both few many much some their them they which what when "
    "where who how because between into through during before after "
    "above below up down out off over under again further then once "
    "here there just only own same other another new old well still "
    "even us our we you your he she him her his".split()
)


def _web_tokenise(text: str) -> set[str]:
    """Extract distinctive lowercase keywords from *text*."""
    tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
    return {t for t in tokens if t not in _STOP}


def annotate_web_citations(
    formatted: str,
    web_results: list[dict[str, str]],
    min_keyword_overlap: int = 3,
    min_overlap_ratio: float = 0.25,
) -> str:
    """Add inline ``[WN]`` citations to formatted analysis lines.

    Works the same way as the literature ``annotate_citations`` but uses
    web-search snippets and the ``[W1]``, ``[W2]``, … prefix so that
    web and literature references are visually distinct.

    Parameters
    ----------
    formatted:
        The formatted analysis text.
    web_results:
        Web search result dicts (each with ``title``, ``snippet``, ``url``).
    min_keyword_overlap:
        Minimum shared keywords for a citation to be added.
    min_overlap_ratio:
        Minimum fraction of the line's keywords that must appear in the
        web snippet.

    Returns
    -------
    The text with ``[WN]`` citations appended to matching lines.
    """
    if not web_results:
        return formatted

    # Build keyword sets for each web snippet
    snippet_keywords: list[set[str]] = []
    for r in web_results:
        combined = (r.get("title", "") + " " + r.get("snippet", "")).strip()
        snippet_keywords.append(_web_tokenise(combined))

    _skip_prefixes = (
        "COUPLING CLASSIFICATION", "SYSTEMS IDENTIFICATION",
        "FLOWS ANALYSIS", "AGENTS", "CAUSES", "EFFECTS",
        "RESEARCH GAPS", "PERICOUPLING DATABASE",
        "METACOUPLING FRAMEWORK", "SUPPORTING EVIDENCE",
        "RECOMMENDED LITERATURE", "WEB SOURCES", "Note:",
    )

    lines = formatted.split("\n")
    annotated: list[str] = []

    in_ref_block = False

    for line in lines:
        stripped = line.strip()

        # Don't annotate inside reference blocks (evidence or web sources)
        if any(h in stripped for h in (
            "SUPPORTING EVIDENCE FROM LITERATURE",
            "WEB SOURCES",
        )):
            in_ref_block = True
        if in_ref_block:
            annotated.append(line)
            continue

        # Skip empty lines, separators, and section headers
        if (
            not stripped
            or stripped.startswith(tuple(_skip_prefixes))
            or set(stripped) <= {"=", "-", " "}
        ):
            annotated.append(line)
            continue

        # Skip sub-headings
        if stripped.startswith("[") and stripped.endswith("]"):
            annotated.append(line)
            continue
        if stripped.endswith(":") and len(stripped.split()) <= 3:
            annotated.append(line)
            continue

        line_tokens = _web_tokenise(stripped)
        if len(line_tokens) < 3:
            annotated.append(line)
            continue

        # Find matching web snippets
        citations: list[int] = []
        for idx, sk in enumerate(snippet_keywords):
            overlap = line_tokens & sk
            n_overlap = len(overlap)
            ratio = n_overlap / len(line_tokens) if line_tokens else 0
            if n_overlap >= min_keyword_overlap and ratio >= min_overlap_ratio:
                citations.append(idx + 1)

        if citations:
            # Avoid duplicates if LLM already cited [WN] in the line
            existing = set()
            for m in re.finditer(r"\[W(\d+)\]", line):
                existing.add(int(m.group(1)))
            new_cites = [c for c in citations if c not in existing]
            if new_cites:
                cite_str = " " + " ".join(f"[W{n}]" for n in new_cites)
                annotated.append(f"{line}{cite_str}")
            else:
                annotated.append(line)
        else:
            annotated.append(line)

    return "\n".join(annotated)
