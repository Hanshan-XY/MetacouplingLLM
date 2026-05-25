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
from dataclasses import dataclass, field
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
        """Return search results as ``title``/``model_summary``/``url`` dicts."""
        ...


# JSON schema for OpenAI ``text.format.json_schema`` strict mode on
# the web-search call.  When supplied, OpenAI guarantees the response
# conforms to this shape, removing the need for prompt-based JSON
# extraction.  See ``OpenAIWebSearchBackend.search()``.
_OPENAI_WEB_SEARCH_RESULTS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "model_summary": {"type": "string"},
                },
                "required": ["title", "url", "model_summary"],
            },
        },
    },
    "required": ["results"],
}


# PR #28: user-defined tool used by AnthropicWebSearchBackend to coax
# Claude into returning a strict-JSON result list AFTER it finishes
# its web_search invocations.  Anthropic docs don't show an
# end-to-end example of ``output_config.format`` json_schema combined
# with a server tool like ``web_search``, so the canonical pattern
# is to define a user-defined tool with ``strict: True`` and instruct
# the model to call it as its final action of the turn.  The schema
# mirrors ``_OPENAI_WEB_SEARCH_RESULTS_SCHEMA`` so both backends
# produce the same downstream shape.
_ANTHROPIC_WEB_SEARCH_SUBMIT_RESULTS_TOOL: dict[str, object] = {
    "name": "submit_results",
    "description": (
        "Submit the final structured list of web search results.  "
        "Call this tool AFTER you have completed all the web_search "
        "invocations needed for the research question.  Each result "
        "must include `title`, `url`, and a 200-400 word "
        "`model_summary` grounded in the actual web search results."
    ),
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string"},
                        "url": {"type": "string"},
                        "model_summary": {"type": "string"},
                    },
                    "required": ["title", "url", "model_summary"],
                },
            },
        },
        "required": ["results"],
    },
    "strict": True,
}


# PR #29: schema used by GrokWebSearchBackend's strict json_schema
# response_format.  xAI's /responses endpoint does NOT return
# server_side tool result blocks separately (web_search runs
# server-side and only the final assistant message is returned),
# so citations must be embedded in the structured response itself.
# That's the difference vs the OpenAI / Anthropic shape: Grok
# requires an evidence_urls field on each result (the URLs the
# model used to write the model_summary), enabling caller-side
# citation tracking without a separate parser.
_GROK_WEB_SEARCH_RESULTS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "model_summary": {"type": "string"},
                    # PR #29 -- Grok-specific: per-result citation
                    # URLs the model consulted when writing the
                    # summary.  Empty list when the result was
                    # synthesised from a single source (the result's
                    # own ``url``).
                    "evidence_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title", "url", "model_summary", "evidence_urls",
                ],
            },
        },
    },
    "required": ["results"],
}


@dataclass
class OpenAIWebSearchBackend:
    """Web-search backend powered by OpenAI Responses API ``web_search``.

    Results are normalized to the same ``title``/``model_summary``/``url`` shape
    used by the existing DuckDuckGo backends so downstream consumers do not
    need to change.
    """

    client: Any
    model: str = "gpt-5"
    reasoning: str = "default"
    external_web_access: bool = True
    allowed_domains: list[str] | None = None
    # Default blocklist for low-quality / SEO-dominant sources.
    # Common pattern across research-quality web search workflows;
    # users who want everything can pass ``blocked_domains=[]`` or
    # supply their own list.
    blocked_domains: list[str] | None = field(
        default_factory=lambda: ["reddit.com", "quora.com", "pinterest.com"]
    )
    # OpenAI web_search tool config.  ``search_context_size`` controls
    # how much context from web results is made available to the model
    # ("low" | "medium" | "high"); default "high" maximises evidence
    # depth at modest token cost.  ``return_token_budget`` controls the
    # output token budget for GPT-5+ reasoning models ("default" |
    # "unlimited"); we default to "default" so callers opt into the
    # unbounded path explicitly.
    search_context_size: str = "high"
    return_token_budget: str = "default"
    user_location: dict[str, object] | None = None
    # Defensive ceiling on the API response token count.  Strict
    # json_schema mode keeps output compact, but a runaway model with
    # 10+ rich evidence summaries could still grow large.  The value
    # acts as a FLOOR -- ``search()`` raises it to ``max_results * 1000``
    # when more results are requested (see effective_max_tokens
    # calculation in ``search()``).  12000 covers ~10-12 results at
    # 200-400 word summaries with ~500 tokens of overhead per result.
    max_output_tokens: int = 12000

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
            "search_context_size": self.search_context_size,
            "return_token_budget": self.return_token_budget,
        }
        # OpenAI web_search tool accepts a ``filters`` object with
        # ``allowed_domains`` and ``blocked_domains``.  Mirror the
        # existing Anthropic backend's signature so both adapters
        # offer the same domain-control surface.  Gemini and Grok
        # don't have native blocklists -- those would need
        # post-filtering and are deferred.
        if self.allowed_domains or self.blocked_domains:
            filters: dict[str, list[str]] = {}
            if self.allowed_domains:
                filters["allowed_domains"] = self.allowed_domains
            if self.blocked_domains:
                filters["blocked_domains"] = self.blocked_domains
            tool["filters"] = filters
        if self.user_location:
            tool["user_location"] = self.user_location

        prompt = (
            "You are a web research collector.\n\n"
            "Your job: find authoritative, diverse, and relevant web "
            "sources for the research question below.  A separate "
            "downstream step will extract structured evidence cards "
            "from your output -- your job is just to return the "
            "sources cleanly, with model-written summaries that "
            "capture the substance.\n\n"
            "Do NOT answer the user's research question.\n"
            "Do NOT write a prose report or analysis.\n"
            "Do NOT include any information not grounded in the web "
            "search results.\n\n"
            "Source selection rules:\n"
            "- Prefer primary or high-authority sources: peer-reviewed "
            "papers, government pages, international organizations, "
            "reputable NGOs, official datasets, and reputable "
            "journalism.\n"
            "- Include diverse perspectives when the question has "
            "multiple dimensions.\n"
            "- Avoid low-quality SEO pages, content farms, forums, "
            "Reddit, Quora, Pinterest, and duplicated syndicated "
            "articles.\n"
            "- Avoid near-duplicate sources that make the same point "
            "from the same underlying report.\n"
            "- Prefer recent sources for current facts, but include "
            "older foundational sources when they remain important.\n"
            "- If fewer than the requested number of strong sources "
            "are available, return fewer.  Do not pad the output.\n\n"
            "Grounding rules:\n"
            "- Use only information found through web_search.\n"
            "- Do not invent URLs, titles, authors, publication "
            "dates, organizations, or findings.\n"
            "- Do not claim that a source supports something unless "
            "the source actually supports it.\n"
            "- Each `model_summary` should be 200-400 words: capture "
            "the source's key findings, specific numeric facts or "
            "quotes when relevant, and the geographic/temporal "
            "scope.  Concise is okay for off-topic results.\n"
            "- `model_summary` is the field name in the output "
            "schema.  Each summary is model-written prose, NOT a "
            "verbatim excerpt.\n\n"
            f"Research question:\n{query.strip()}\n\n"
            f"Return up to {max_results} highly relevant results as "
            "JSON conforming to the provided schema."
        )

        # Scale the response token ceiling with the requested result
        # count: 1000 tokens/result is a generous safety margin (a
        # 400-word summary is ~500 tokens; another ~500 covers
        # title/URL/JSON overhead per result).  The dataclass default
        # acts as a floor so callers can still raise it further.
        # Without this scaling, ``max_results > ~10`` truncates the
        # model's JSON output, ``_extract_json_object`` fails, and
        # ``parsed_results`` falls back to URL-only source citations
        # with empty summaries -- the avocado-25-results bug
        # documented in PR #24.
        effective_max_tokens = max(
            self.max_output_tokens,
            max_results * 1000,
        )

        kwargs: dict[str, object] = {
            "model": self.model,
            "tools": [tool],
            # ``tool_choice="required"`` forces the model to invoke
            # web_search.  Our whole purpose here IS to search; "auto"
            # would let the model skip the tool and answer from
            # training data, defeating the call.
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "input": prompt,
            # Strict json_schema mode: OpenAI guarantees the response
            # conforms to ``_OPENAI_WEB_SEARCH_RESULTS_SCHEMA``, which
            # makes the ``_extract_json_object`` fallback below largely
            # decorative for OpenAI.  We keep the fallback for cases
            # where adapters wrapping the SDK strip the format field
            # or where the model still emits warnings inside the
            # output_text envelope.
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "web_search_results",
                    "strict": True,
                    "schema": _OPENAI_WEB_SEARCH_RESULTS_SCHEMA,
                },
            },
            # Defensive ceiling on response token count.  See field
            # docstring on the dataclass.
            "max_output_tokens": effective_max_tokens,
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

        # Warn on partial truncation: the model returned SOME parsed
        # results but fewer than half of what we requested.  This
        # usually means the JSON output was cut off mid-array.
        if (
            parsed_results
            and len(parsed_results) < max(1, int(max_results * 0.5))
        ):
            print(
                f"[OpenAIWebSearchBackend] Warning: parsed "
                f"{len(parsed_results)}/{max_results} results -- "
                "model may have truncated summary content (try "
                "raising max_output_tokens)."
            )

        # Fallback: if the model didn't emit JSON results cleanly, recover
        # from the included source list so the search still yields usable URLs.
        source_results = _extract_openai_source_results(response, max_results)

        # Warn on total fallback: parsed JSON was unusable AND we're
        # only returning URL-only source citations (titles == urls,
        # no model_summary content).  Downstream Stage-1 extraction
        # will see empty summaries and silently emit zero receivers.
        if not parsed_results and source_results:
            print(
                f"[OpenAIWebSearchBackend] Warning: parsed 0/"
                f"{max_results} results from JSON; falling back to "
                "URL-only source citations (model summaries lost; "
                "downstream extraction will have no grounded data)."
            )

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
    claude-sonnet-4-6 (default) web_search_20260209
    claude-opus-4-7             web_search_20260209
    claude-opus-4-6             web_search_20260209
    claude-opus-4-5 / 4-1 / 4-0 web_search_20250305
    claude-sonnet-4-5 / 4-0     web_search_20250305
    claude-haiku-4-5            web_search_20250305
    unknown                     web_search_20260209
    ==========================  ==========================

    Pass an explicit ``tool_version`` to override (useful when a new model
    ships before the lookup table is updated).

    Result extraction (PR #28): the primary path is the
    ``submit_results`` user-defined tool -- the system prompt asks
    Claude to call this tool with a structured list of
    ``title``/``url``/``model_summary`` entries as the final action
    of its turn.  ``_extract_anthropic_submit_results_tool_use``
    walks the response for that tool_use block.  When Claude
    ignores the instruction (rare), the backend logs a warning and
    falls back to ``_extract_anthropic_web_results`` which mines
    citations on text blocks (each citation carries ``url``,
    ``title``, and up to 150 chars of ``cited_text``) and then the
    raw ``web_search_tool_result`` blocks for ``url`` + ``title``
    + ``page_age`` (``encrypted_content`` is opaque to callers by
    design).

    Prompt caching is intentionally not used: a web-search call's cacheable
    prefix (tool definition plus short user prompt) is well under the
    Sonnet 4.6 / Opus 4.6+ 4096-token minimum, so caching would
    silently no-op.
    """

    client: Any
    # PR #28: default bumped from claude-opus-4-7 to claude-sonnet-4-6
    # for cost.  Sonnet 4.6 still supports web_search_20260209
    # (dynamic filtering) per Anthropic docs.  Pricing: $3/$15 per
    # MTok vs Opus 4.7's $5/$25.  Override via the ``model`` kwarg
    # if you need Opus.
    model: str = "claude-sonnet-4-6"
    tool_version: str | None = None   # None = auto-select from model
    max_uses: int = 5
    allowed_domains: list[str] | None = None
    # PR #28: default blocklist mirrors OpenAIWebSearchBackend (PR #18).
    # Users who want everything can pass ``blocked_domains=[]`` or
    # supply their own list.
    blocked_domains: list[str] | None = field(
        default_factory=lambda: ["reddit.com", "quora.com", "pinterest.com"]
    )
    user_location: dict[str, object] | None = None
    # PR #28: bumped 8192 -> 12000 to match OpenAIWebSearchBackend
    # (PR #24).  Acts as a *floor* -- ``search()`` raises it to
    # ``max_results * 1000`` when more results are requested so the
    # model has room for 25+ rich summaries without truncation.
    max_tokens: int = 12000

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        # PR #28: default to BASIC ``web_search_20250305`` instead of
        # auto-inferring from the model.  The dynamic-filtering tool
        # (``web_search_20260209``) requires pairing with
        # ``code_execution``, and the combination caused Claude to
        # bypass our ``submit_results`` tool in the live avocado-25
        # trace -- the bypass degrades output from rich 200-400 word
        # ``model_summary`` blocks to bare title+url+page_age strings.
        # Basic web_search lets Claude either call submit_results
        # naturally OR write text with citations (~150 chars each),
        # both significantly richer than the dynamic-filtering
        # fallback path.  Users who need dynamic filtering can opt
        # in explicitly by passing
        # ``tool_version="web_search_20260209"`` to the constructor
        # (the ``_infer_web_search_tool_version`` lookup helper is
        # retained for that case).
        tool_version = self.tool_version or "web_search_20250305"
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

        # PR #28: tools list = web_search + submit_results.  When the
        # caller explicitly opts in to dynamic filtering by passing
        # tool_version="web_search_20260209", we also include the
        # code_execution_20260120 tool that Anthropic requires for
        # that mode.  Default path uses basic web_search only and
        # skips code_execution entirely.
        tools_list: list[dict[str, object]] = [
            tool,
            _ANTHROPIC_WEB_SEARCH_SUBMIT_RESULTS_TOOL,
        ]
        has_dynamic_filtering = tool_version == "web_search_20260209"
        if has_dynamic_filtering:
            tools_list.append({
                "type": "code_execution_20260120",
                "name": "code_execution",
            })

        # PR #28: rich prompt template mirroring OpenAI PR #18.
        # Same role / DO-NOTs / source-selection / grounding rules.
        # Closing instruction adapted from OpenAI's "Return JSON
        # conforming to the schema" to "Call the submit_results
        # tool" since Claude returns structured data via tool_use
        # rather than json_schema'd text.
        prompt = (
            "You are a web research collector.\n\n"
            "Your job: find authoritative, diverse, and relevant web "
            "sources for the research question below.  A separate "
            "downstream step will extract structured evidence cards "
            "from your output -- your job is just to return the "
            "sources cleanly, with model-written summaries that "
            "capture the substance.\n\n"
            "Do NOT answer the user's research question.\n"
            "Do NOT write a prose report or analysis.\n"
            "Do NOT include any information not grounded in the web "
            "search results.\n\n"
            "Source selection rules:\n"
            "- Prefer primary or high-authority sources: peer-reviewed "
            "papers, government pages, international organizations, "
            "reputable NGOs, official datasets, and reputable "
            "journalism.\n"
            "- Include diverse perspectives when the question has "
            "multiple dimensions.\n"
            "- Avoid low-quality SEO pages, content farms, forums, "
            "Reddit, Quora, Pinterest, and duplicated syndicated "
            "articles.\n"
            "- Avoid near-duplicate sources that make the same point "
            "from the same underlying report.\n"
            "- Prefer recent sources for current facts, but include "
            "older foundational sources when they remain important.\n"
            "- If fewer than the requested number of strong sources "
            "are available, return fewer.  Do not pad the output.\n\n"
            "Grounding rules:\n"
            "- Use only information found through web_search.\n"
            "- Do not invent URLs, titles, authors, publication "
            "dates, organizations, or findings.\n"
            "- Do not claim that a source supports something unless "
            "the source actually supports it.\n"
            "- Each `model_summary` should be 200-400 words: capture "
            "the source's key findings, specific numeric facts or "
            "quotes when relevant, and the geographic/temporal "
            "scope.  Concise is okay for off-topic results.\n"
            "- `model_summary` is the field name in the "
            "submit_results tool's input schema.  Each summary is "
            "model-written prose, NOT a verbatim excerpt.\n\n"
            f"Research question:\n{query.strip()}\n\n"
            f"Return up to {max_results} highly relevant results.\n\n"
            "**REQUIRED FINAL ACTION**: After completing your "
            "web_search invocations, you MUST call the "
            "`submit_results` tool with the structured findings.  "
            "Do NOT end your turn with just text or tool_results -- "
            "the only valid way to complete this task is to call "
            "submit_results.  If you have finished searching, call "
            "submit_results NOW; do not stop without calling it."
        )

        # PR #28: scale the model's output budget with max_results
        # (same formula as OpenAI PR #24).  1000 tokens/result is a
        # generous margin for a 400-word summary plus structure
        # overhead.
        effective_max_tokens = max(self.max_tokens, max_results * 1000)

        # PR #28: force the first turn to actually invoke web
        # research rather than answering from training data
        # (Anthropic analogue of OpenAI's tool_choice="required").
        # When dynamic filtering is on, web_search_20260209 is NOT
        # directly callable by the model -- it can only be invoked
        # from inside code_execution.  The live trace surfaced this
        # via an API 400: "tool_choice.name 'web_search' cannot be
        # used because this tool only allows calls from
        # ['code_execution_20260120']".  In that mode we tool_choice
        # code_execution instead; on the basic web_search_20250305
        # (no dynamic filtering) the model calls web_search
        # directly.
        if has_dynamic_filtering:
            forced_tool_name = "code_execution"
        else:
            forced_tool_name = "web_search"

        # PR #28: ALWAYS stream for web search.  Anthropic's SDK
        # refuses non-streaming requests whose estimated generation
        # time would exceed 10 minutes, but the upfront refusal is
        # calibrated only to ``max_tokens`` -- it can't see the
        # wall-clock cost of server-side web_search + code_execution
        # sub-calls.  Even small max_tokens calls can take >10 min
        # when Claude dispatches multiple sub-searches with dynamic
        # filtering, in which case non-streaming would hang or
        # error mid-call.  Streaming is the same price, and
        # ``messages.stream(...).get_final_message()`` returns the
        # same ``Message`` shape as ``messages.create()``, so the
        # downstream parsing path is unchanged regardless of
        # request size.  The threshold-based pattern from
        # ``AnthropicAdapter`` (text-only chat) doesn't transfer
        # here because plain chat has predictable max_tokens-bound
        # wall-clock while server tools don't.
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "tools": tools_list,
            "tool_choice": {"type": "tool", "name": forced_tool_name},
            "messages": [{"role": "user", "content": prompt}],
        }
        # PR #28: retry on transient streaming-connection failures.
        # Anthropic's SDK does NOT retry streaming requests
        # automatically (unlike messages.create()), so a single
        # network blip or server-side timeout closes the chunked
        # connection mid-stream and surfaces as a "peer closed
        # connection without sending complete message body" /
        # "incomplete chunked read" / APIConnectionError /
        # RemoteProtocolError.  The 5-probe diagnostic at
        # scripts/diagnose_anthropic_web_search.py confirmed the
        # PR #28 config (web_search_20260209 + code_execution +
        # submit_results + max_tokens=25000 + max_uses=5) is
        # structurally correct -- failures observed in the live
        # avocado trace were transient.  Try up to 3 times with
        # exponential backoff (2s, 4s) before propagating.
        import time as _time
        _import_failed = False
        try:
            import anthropic as _anthropic
            import httpx as _httpx
            _transient_exc_types: tuple[type[Exception], ...] = (
                _anthropic.APIConnectionError,
                _httpx.ReadError,
                _httpx.RemoteProtocolError,
            )
        except Exception:
            # Defensive: if anthropic/httpx imports fail at runtime
            # for some test stub reason, fall back to catching any
            # IOError-style exception so retries still run.
            _transient_exc_types = (OSError,)
            _import_failed = True

        last_exc: Exception | None = None
        response = None
        for attempt in range(3):
            try:
                with self.client.messages.stream(**request_kwargs) as stream:
                    response = stream.get_final_message()
                break
            except _transient_exc_types as exc:
                last_exc = exc
                if attempt < 2:
                    print(
                        "[AnthropicWebSearchBackend] Warning: streaming "
                        f"connection failed ({type(exc).__name__}); "
                        f"retrying in {2 * (2 ** attempt)}s "
                        f"({attempt + 1}/3)..."
                    )
                    _time.sleep(2 * (2 ** attempt))
                    continue
                # All retries exhausted -- propagate so the upstream
                # MetacouplingAssistant catches it and falls back to
                # DuckDuckGo (existing behaviour).
                raise
        if response is None:  # pragma: no cover -- defensive
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "AnthropicWebSearchBackend: stream returned no response"
            )

        # PR #28: try the strict-JSON submit_results path first.
        # When Claude obeys the instruction and calls the tool, the
        # results come back fully structured.  When it doesn't, fall
        # back to the citation/tool_result parser and print a
        # diagnostic warning (silent-fallback parity with OpenAI's
        # PR #24).
        submit_results = _extract_anthropic_submit_results_tool_use(
            response, max_results,
        )
        if submit_results:
            return submit_results

        print(
            "[AnthropicWebSearchBackend] Warning: Claude did not call "
            "submit_results; falling back to citation-based result "
            "extraction (structured output may be incomplete)."
        )
        return _extract_anthropic_web_results(response, max_results)


@dataclass
class GeminiWebSearchBackend:
    """Web-search backend powered by Google Gemini's grounding tool.

    PR #29: brought to parity with OpenAI (PRs #17/#18/#24) and
    Anthropic (PR #28) -- rich prompt, strict structured output,
    blocklist (client-side post-hoc), max_tokens scaling,
    silent-fallback warning, Gemini 3.1 Pro as default.

    Built against the new ``google.genai`` SDK.  Calls
    ``client.models.generate_content`` with both the ``google_search``
    tool AND a strict ``response_schema`` (Gemini natively supports
    combining grounding + structured output in one call; no
    submit_results dance like Anthropic).  Model performs the search,
    synthesises 200-400 word per-source summaries, and emits a
    schema-conformant JSON object.  ``_extract_gemini_grounding_results``
    is kept as a fallback parser for grounding metadata when the
    structured response is empty or malformed.

    Parameters
    ----------
    client:
        A ``google.genai.Client`` instance.
    model:
        Gemini model identifier supporting grounding (default
        ``"gemini-3.1-pro-preview"``; still in preview per Google's
        official changelog as of mid-2026 -- switch to
        ``"gemini-3.1-pro"`` when GA lands).  Other 3.x variants:
        ``"gemini-3.1-flash-lite"`` (cheaper, less reasoning).
    allowed_domains / blocked_domains:
        Post-hoc URL filters on the structured response.  Gemini's
        ``google_search`` tool does NOT support server-side domain
        filtering (could not confirm any 2026 addition from docs);
        we apply the filter client-side as a parity belt against
        the OpenAI / Anthropic backends.  Default blocked_domains
        mirror OpenAIWebSearchBackend (PR #18).
    thinking_level:
        Gemini 3.1 Pro forces extended thinking ON regardless of
        client config; this parameter controls the *budget*.  Values:
        ``"minimal" | "low" | "medium" | "high"``.  Default
        ``"high"`` per user preference (cost is not a constraint;
        optimise for quality).
    max_tokens:
        Output budget for the search-and-summarise call.  Acts as a
        floor; ``search()`` raises it to ``max_results * 1000``
        when more results are requested, capped at Gemini 3.1 Pro's
        64k output ceiling.
    """

    client: Any
    # PR #29: default bumped from gemini-2.5-flash to
    # gemini-3.1-pro-preview per user request ("latest Gemini 3.1
    # pro").  Still preview per Google's official changelog as of
    # 2026-05; switch to "gemini-3.1-pro" once GA confirmed.
    model: str = "gemini-3.1-pro-preview"
    # PR #29: parity defaults mirroring OpenAI/Anthropic.
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = field(
        default_factory=lambda: ["reddit.com", "quora.com", "pinterest.com"]
    )
    # PR #29: thinking budget.  Gemini 3.1 Pro can't be made to
    # skip thinking entirely; this controls how many thinking
    # tokens the model spends.  "high" per user memory preference.
    thinking_level: str = "high"
    # PR #29: bumped 8192 -> 12000 to match OpenAI/Anthropic.
    # Acts as a floor; per-call scales to ``max_results * 1000``
    # capped at Gemini 3.1 Pro's 64k output ceiling.
    max_tokens: int = 12000

    # PR #29: Gemini 3.1 Pro's hard output cap.
    _MAX_OUTPUT_TOKENS_CEILING: int = 64000

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        # PR #29: rich prompt template mirroring OpenAI PR #18.
        # Same role / DO-NOTs / source-selection / grounding rules.
        # Closing instruction adapted from OpenAI's "Return JSON
        # conforming to the provided schema" to match Gemini's
        # native response_schema enforcement (no separate
        # submit_tool needed -- Gemini supports schema + grounding
        # in one call per
        # https://ai.google.dev/gemini-api/docs/structured-output).
        prompt = (
            "You are a web research collector.\n\n"
            "Your job: find authoritative, diverse, and relevant web "
            "sources for the research question below.  A separate "
            "downstream step will extract structured evidence cards "
            "from your output -- your job is just to return the "
            "sources cleanly, with model-written summaries that "
            "capture the substance.\n\n"
            "Do NOT answer the user's research question.\n"
            "Do NOT write a prose report or analysis.\n"
            "Do NOT include any information not grounded in the "
            "google_search results.\n\n"
            "Source selection rules:\n"
            "- Prefer primary or high-authority sources: peer-reviewed "
            "papers, government pages, international organizations, "
            "reputable NGOs, official datasets, and reputable "
            "journalism.\n"
            "- Include diverse perspectives when the question has "
            "multiple dimensions.\n"
            "- Avoid low-quality SEO pages, content farms, forums, "
            "Reddit, Quora, Pinterest, and duplicated syndicated "
            "articles.\n"
            "- Avoid near-duplicate sources that make the same point "
            "from the same underlying report.\n"
            "- Prefer recent sources for current facts, but include "
            "older foundational sources when they remain important.\n"
            "- If fewer than the requested number of strong sources "
            "are available, return fewer.  Do not pad the output.\n\n"
            "Grounding rules:\n"
            "- Use only information found through google_search.\n"
            "- Do not invent URLs, titles, authors, publication "
            "dates, organizations, or findings.\n"
            "- Do not claim that a source supports something unless "
            "the source actually supports it.\n"
            "- Each `model_summary` should be 200-400 words: capture "
            "the source's key findings, specific numeric facts or "
            "quotes when relevant, and the geographic/temporal "
            "scope.  Concise is okay for off-topic results.\n"
            "- `model_summary` is the field name in the output "
            "schema.  Each summary is model-written prose, NOT a "
            "verbatim excerpt.\n\n"
            f"Research question:\n{query.strip()}\n\n"
            f"Return up to {max_results} highly relevant results as "
            "JSON conforming to the provided response schema."
        )

        # PR #29: scale output budget with max_results, capped at
        # Gemini 3.1 Pro's 64k hard ceiling.
        effective_max_tokens = min(
            max(self.max_tokens, max_results * 1000),
            self._MAX_OUTPUT_TOKENS_CEILING,
        )

        # PR #29: combined google_search + structured response_schema
        # is documented to work together on Gemini 3.x.  Same
        # ``_OPENAI_WEB_SEARCH_RESULTS_SCHEMA`` shape so downstream
        # ``_normalise_backend_results`` consumes all three backends
        # identically.
        config: dict[str, object] = {
            "max_output_tokens": effective_max_tokens,
            "tools": [{"google_search": {}}],
            "response_mime_type": "application/json",
            "response_schema": _OPENAI_WEB_SEARCH_RESULTS_SCHEMA,
            "thinking_config": {"thinking_level": self.thinking_level},
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            print(
                "[GeminiWebSearchBackend] Warning: generate_content "
                f"failed ({type(exc).__name__}: {exc}); returning empty "
                "result list."
            )
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

        # PR #29: post-hoc domain filter (Gemini lacks server-side
        # blocked_domains for google_search; we filter client-side
        # as a parity belt).
        parsed_results = self._apply_domain_filters(parsed_results)

        # Fallback: walk grounding_metadata.grounding_chunks for
        # url + title.  Used when the structured response is
        # missing / malformed.
        grounding_results = _extract_gemini_grounding_results(
            response, max_results
        )
        grounding_results = self._apply_domain_filters(grounding_results)

        if not parsed_results and grounding_results:
            print(
                "[GeminiWebSearchBackend] Warning: structured "
                "response was empty / malformed; falling back to "
                "grounding-metadata extraction (title + url only, "
                "no model_summary)."
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

    def _apply_domain_filters(
        self, results: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Post-hoc client-side blocked_domains / allowed_domains
        filter on result URLs.  PR #29 parity belt for Gemini
        which lacks server-side domain filtering."""
        if not (self.allowed_domains or self.blocked_domains):
            return results
        filtered: list[dict[str, str]] = []
        for item in results:
            url = str(item.get("url", "")).lower()
            if not url:
                continue
            if self.blocked_domains and any(
                blocked in url for blocked in self.blocked_domains
            ):
                continue
            if self.allowed_domains and not any(
                allowed in url for allowed in self.allowed_domains
            ):
                continue
            filtered.append(item)
        return filtered


@dataclass
class GrokWebSearchBackend:
    """Web-search backend powered by xAI Grok's Built-in Tools.

    PR #29 (MAJOR REFACTOR): migrated from the deprecated
    ``chat.completions`` + ``search_parameters`` API (HTTP 410 Gone
    since 2026-01-12) to the new ``/responses`` endpoint with
    server-side ``tools=[{"type": "web_search", ...}]``.  The old
    Live Search API parameters (``search_mode``, ``max_search_results``,
    ``sources``, ``country``, ``language``, ``from_date``, ``to_date``)
    are GONE for web_search and have no replacements.

    Brought to parity with OpenAI (PRs #17/#18/#24) and Anthropic
    (PR #28): rich prompt, strict json_schema response_format,
    excluded_domains (capped at 5 per xAI API), max_tokens scaling,
    silent-fallback warning, Grok 4.3 as default,
    reasoning_effort="high" per user preference.

    Architecture difference vs OpenAI / Anthropic: xAI's
    ``/responses`` server-side tool path does NOT return separate
    tool_result blocks; the model runs the search loop internally
    and returns one final assistant message containing the
    structured JSON.  Citations are required as an ``evidence_urls``
    schema field (see ``_GROK_WEB_SEARCH_RESULTS_SCHEMA``) because
    there's no separate citation parser path.

    Known parity gaps documented:
    - **No ``tool_choice`` forcing analogue**: xAI's ``/responses``
      endpoint doesn't expose one.  Forced-search is expressed via
      prompt language only ("**REQUIRED**: you MUST use the
      web_search tool").
    - **No per-source max_results**: ``max_turns`` caps invocations
      (1-2 quick / 3-5 balanced / 10+ deep).

    Parameters
    ----------
    client:
        An ``openai.OpenAI`` instance configured with
        ``base_url="https://api.x.ai/v1"``.
    model:
        Grok model identifier (default ``"grok-4.3"``).  Other
        variants: ``"grok-4.1-fast"`` (cheaper, 2M ctx).
    excluded_domains / allowed_domains:
        xAI's ``/responses`` ``web_search`` tool accepts up to 5
        domains in each list (mutually exclusive).  Default
        ``excluded_domains`` mirrors OpenAIWebSearchBackend (PR #18).
    reasoning_effort:
        ``"none" | "low" | "medium" | "high"``.  Applies to Grok 4.3.
        Default ``"high"`` per user memory preference.
    max_turns:
        Server-side tool-invocation budget for the search loop
        (1-2 quick / 3-5 balanced / 10+ deep).  Default 5
        (balanced).
    enable_image_understanding:
        Whether the web_search tool should also fetch and process
        images from results.  Default False (text-only, faster).
    max_tokens:
        Output budget for the search-and-summarise call.  Acts as
        a floor; ``search()`` scales to ``max_results * 1000``.
    """

    client: Any
    # PR #29: default bumped grok-3 -> grok-4.3 per user request.
    # Aliases: grok-4.3-latest, grok-latest.
    model: str = "grok-4.3"
    # PR #29: parity defaults.  xAI's web_search tool caps each
    # domain list at 5 entries; the OpenAI/Anthropic 3-entry
    # default fits comfortably.
    allowed_domains: list[str] | None = None
    excluded_domains: list[str] | None = field(
        default_factory=lambda: ["reddit.com", "quora.com", "pinterest.com"]
    )
    # PR #29: "high" per user memory preference (cost is not a
    # constraint; optimise for quality).
    reasoning_effort: str = "high"
    # PR #29: xAI's "balanced" deep-search cap (3-5).
    max_turns: int = 5
    enable_image_understanding: bool = False
    # PR #29: bumped 8192 -> 12000 to match OpenAI/Anthropic.
    # Acts as a floor; per-call scales to ``max_results * 1000``.
    max_tokens: int = 12000

    # PR #29: xAI's web_search tool API hard limit (≤5 domains per
    # allowed/excluded list).  Source: docs.x.ai/developers/tools/web-search.
    _MAX_DOMAIN_LIST_SIZE: int = 5

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        if not query.strip() or max_results <= 0:
            return []

        # PR #29: rich prompt template mirroring OpenAI PR #18.
        # Stronger language ("**REQUIRED**: you MUST use the
        # web_search tool") because xAI has no tool_choice="required"
        # analogue -- forcing happens via prompt only.  Closing
        # instruction matches the strict json_schema response_format
        # below.
        prompt = (
            "You are a web research collector.\n\n"
            "Your job: find authoritative, diverse, and relevant web "
            "sources for the research question below.  A separate "
            "downstream step will extract structured evidence cards "
            "from your output -- your job is just to return the "
            "sources cleanly, with model-written summaries that "
            "capture the substance.\n\n"
            "Do NOT answer the user's research question.\n"
            "Do NOT write a prose report or analysis.\n"
            "Do NOT include any information not grounded in the "
            "web_search tool's results.\n"
            "**REQUIRED**: you MUST use the web_search tool to "
            "gather sources; do not answer from training data.\n\n"
            "Source selection rules:\n"
            "- Prefer primary or high-authority sources: peer-reviewed "
            "papers, government pages, international organizations, "
            "reputable NGOs, official datasets, and reputable "
            "journalism.\n"
            "- Include diverse perspectives when the question has "
            "multiple dimensions.\n"
            "- Avoid low-quality SEO pages, content farms, forums, "
            "Reddit, Quora, Pinterest, and duplicated syndicated "
            "articles.\n"
            "- Avoid near-duplicate sources that make the same point "
            "from the same underlying report.\n"
            "- Prefer recent sources for current facts, but include "
            "older foundational sources when they remain important.\n"
            "- If fewer than the requested number of strong sources "
            "are available, return fewer.  Do not pad the output.\n\n"
            "Grounding rules:\n"
            "- Use only information found through web_search.\n"
            "- Do not invent URLs, titles, authors, publication "
            "dates, organizations, or findings.\n"
            "- Do not claim that a source supports something unless "
            "the source actually supports it.\n"
            "- Each `model_summary` should be 200-400 words: capture "
            "the source's key findings, specific numeric facts or "
            "quotes when relevant, and the geographic/temporal "
            "scope.  Concise is okay for off-topic results.\n"
            "- `model_summary` is the field name in the output "
            "schema.  Each summary is model-written prose, NOT a "
            "verbatim excerpt.\n"
            "- `evidence_urls` lists the URLs you actually used to "
            "write each `model_summary` -- typically just the "
            "result's own URL, but include any additional URLs "
            "consulted via web_search for that summary.\n\n"
            f"Research question:\n{query.strip()}\n\n"
            f"Return up to {max_results} highly relevant results "
            "conforming exactly to the provided JSON schema."
        )

        # PR #29: scale output budget with max_results (Grok 4.3
        # has no documented hard ceiling).
        effective_max_tokens = max(self.max_tokens, max_results * 1000)

        # PR #29: build the web_search tool spec.  excluded_domains
        # and allowed_domains are capped at 5 per xAI API; we
        # silently truncate if the user passed more.
        web_search_tool: dict[str, object] = {
            "type": "web_search",
            "enable_image_understanding": (
                self.enable_image_understanding
            ),
        }
        if self.excluded_domains:
            web_search_tool["excluded_domains"] = list(
                self.excluded_domains
            )[: self._MAX_DOMAIN_LIST_SIZE]
        if self.allowed_domains:
            web_search_tool["allowed_domains"] = list(
                self.allowed_domains
            )[: self._MAX_DOMAIN_LIST_SIZE]

        # PR #29: strict json_schema response_format combined with
        # web_search tool (explicitly documented to work together
        # at https://docs.x.ai/docs/guides/structured-outputs).
        request_kwargs: dict[str, object] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": effective_max_tokens,
            "tools": [web_search_tool],
            "max_turns": self.max_turns,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "web_search_results",
                    "strict": True,
                    "schema": _GROK_WEB_SEARCH_RESULTS_SCHEMA,
                },
            },
        }

        try:
            response = self.client.responses.create(**request_kwargs)
        except Exception as exc:
            print(
                "[GrokWebSearchBackend] Warning: responses.create "
                f"failed ({type(exc).__name__}: {exc}); returning "
                "empty result list."
            )
            return []

        # PR #29: pull the structured output text from the response.
        # /responses returns ``output_text`` (concatenated assistant
        # text) when response_format is json_schema, similar to
        # OpenAI's Responses API.
        output_text = getattr(response, "output_text", "") or ""
        raw_obj = _extract_json_object(output_text)

        parsed_results: list[dict[str, str]] = []
        if isinstance(raw_obj, dict):
            parsed_results = _normalise_backend_results(
                raw_obj.get("results", []),
                max_results=max_results,
            )

        # PR #29: silent-fallback warning -- if no parsed results
        # came back, the model likely either didn't call web_search
        # (no tool_choice forcing available; prompt is our only
        # lever) or returned a structured output that broke the
        # schema.  Diagnostic parity with OpenAI PR #24 and
        # Anthropic PR #28.
        if not parsed_results:
            # Try to detect whether web_search was even invoked, to
            # give a more actionable warning.
            tool_usage = getattr(response, "server_side_tool_usage", None)
            web_search_calls = 0
            if tool_usage is not None:
                web_search_calls = getattr(
                    tool_usage, "web_search_count", 0,
                ) or 0
            if web_search_calls == 0:
                print(
                    "[GrokWebSearchBackend] Warning: web_search was "
                    "NOT invoked (server_side_tool_usage."
                    "web_search_count == 0); model answered from "
                    "training data.  Strengthen the prompt's "
                    "REQUIRED-tool instruction."
                )
            else:
                print(
                    "[GrokWebSearchBackend] Warning: web_search ran "
                    f"({web_search_calls} call(s)) but structured "
                    "output was empty/malformed; falling through to "
                    "empty result list."
                )

        # Fallback: harvest citations from response.citations if
        # present (older xAI clients still expose this field).
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
            out.append({"title": title, "model_summary": "", "url": url})
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
            summary = ""
        elif isinstance(cite, dict):
            url = str(cite.get("url") or "").strip()
            title = str(cite.get("title") or url).strip()
            # Accept legacy "snippet" key on the way IN; emit
            # "model_summary" on the way OUT.
            summary = str(
                cite.get("model_summary")
                or cite.get("snippet")
                or cite.get("excerpt")
                or ""
            ).strip()
        else:
            url = str(getattr(cite, "url", "") or "").strip()
            title = str(getattr(cite, "title", url) or url).strip()
            summary = str(
                getattr(cite, "model_summary", None)
                or getattr(cite, "snippet", "")
                or ""
            ).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        out.append({"title": title, "model_summary": summary, "url": url})
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
        # ``_in_summary`` tracks the DDG result-snippet <td> -- the
        # CSS class itself is "result-snippet" on the upstream HTML
        # (that's DDG's class name; not ours), but the value we
        # extract is mapped onto our ``model_summary`` output key
        # for consistency with the rest of the backend pipeline.
        self._in_summary = False
        self._current: dict[str, str] = {}
        self._text_buf: list[str] = []

    # DuckDuckGo Lite wraps each result link in
    # <a rel="nofollow" class="result-link" href="…">
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("class") == "result-link":
            self._in_link = True
            self._current = {
                "url": attr.get("href", ""),
                "title": "",
                "model_summary": "",
            }
            self._text_buf = []
        elif tag == "td" and attr.get("class") == "result-snippet":
            self._in_summary = True
            self._text_buf = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            self._current["title"] = " ".join(self._text_buf).strip()
        elif tag == "td" and self._in_summary:
            self._in_summary = False
            self._current["model_summary"] = " ".join(self._text_buf).strip()
            if self._current.get("url"):
                self.results.append(self._current)
            self._current = {}

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_summary:
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
            # DDG's ``body`` field maps onto our ``model_summary``
            # output key.  DDG bodies are verbatim page excerpts
            # rather than model-written summaries, but the field
            # name reflects the shape of the contract, not the
            # provenance.
            "model_summary": r.get("body", ""),
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
                "model_summary": r.get("body", ""),
                "url": _resolve_ddg_url(r.get("href", "")),
            })
    return results


def _result_key(result: dict[str, str]) -> tuple[str, str]:
    """Build a stable dedupe key for a search result."""
    url = result.get("url", "").strip().lower().rstrip("/")
    if url:
        return ("url", url)
    title = " ".join(result.get("title", "").lower().split())
    summary = " ".join(result.get("model_summary", "").lower().split())
    return ("text", f"{title}|{summary}")


def _normalise_backend_results(
    results: object,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    """Normalise backend output to ``title``/``model_summary``/``url`` dicts.

    Accepts a legacy ``snippet`` key from provider responses and maps
    it onto ``model_summary`` so a half-migrated provider doesn't lose
    its summary text.  The output dict ALWAYS uses ``model_summary``;
    callers downstream of this function never see ``snippet``.
    """
    if not isinstance(results, list):
        return []

    normalised: list[dict[str, str]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        # Legacy alias: providers that still emit "snippet" get mapped
        # to "model_summary" silently.  Remove this shim in a future
        # release once all internal providers have migrated.
        model_summary = str(
            item.get("model_summary")
            or item.get("snippet", "")
        ).strip()
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        normalised.append({
            "title": title,
            "model_summary": model_summary,
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
                        "model_summary": str(
                            node.get("model_summary")
                            or node.get("snippet")
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


def _extract_anthropic_submit_results_tool_use(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Extract structured results from Claude's ``submit_results`` tool_use.

    PR #28: when ``AnthropicWebSearchBackend`` includes the
    ``submit_results`` user-defined tool in its tools list and
    instructs Claude to call it after web_search, the structured
    results land in a ``tool_use`` block with
    ``name == "submit_results"``.  This helper walks the response
    content blocks, finds that tool_use, and normalises its
    ``input.results`` array into the standard
    ``title``/``model_summary``/``url`` shape.

    Returns an empty list when no ``submit_results`` tool_use is
    found -- the caller then falls back to
    ``_extract_anthropic_web_results`` (citation-based parser) and
    logs a diagnostic warning.
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

    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        if block.get("name") != "submit_results":
            continue
        block_input = block.get("input")
        if not isinstance(block_input, dict):
            continue
        results = block_input.get("results")
        if not isinstance(results, list):
            continue
        # Hand off to the same normalisation path used by the
        # OpenAI backend -- enforces title/model_summary/url shape,
        # dedupes empty URLs, applies the max_results cap.
        return _normalise_backend_results(results, max_results=max_results)

    return []


def _extract_anthropic_web_results(
    response: object,
    max_results: int,
) -> list[dict[str, str]]:
    """Extract ``title``/``model_summary``/``url`` rows from an Anthropic response.

    Primary path: walk the ``citations`` lists on each ``text`` content
    block — every citation carries the source URL, title, and up to 150
    chars of ``cited_text``.  Dedupe by URL, take the first citation's
    ``cited_text`` as the ``model_summary`` when a URL is cited multiple
    times.  Note: Anthropic's ``cited_text`` is a VERBATIM excerpt
    rather than a model-written summary, so the ``model_summary`` field
    in Anthropic-backend outputs is actually a true quote (~150 chars);
    the field name reflects the dict-shape contract, not the
    provenance of the text.

    Fallback path: when Claude returned search results but no citations,
    walk the ``web_search_tool_result`` blocks for ``url`` + ``title``.
    ``encrypted_content`` is opaque by design, so ``model_summary`` in
    this path degrades to just the ``page_age`` timestamp.
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
            model_summary = str(cit.get("cited_text", "") or "").strip()
            found.append({
                "title": title,
                "model_summary": model_summary,
                "url": url,
            })
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
                model_summary = (
                    f"(page age: {page_age})" if page_age else ""
                )
                found.append({
                    "title": title,
                    "model_summary": model_summary,
                    "url": url,
                })
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
    A list of dicts, each with ``title``, ``model_summary``, and ``url`` keys.
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


def format_web_context(
    results: list[dict[str, str]], turn: int = 1
) -> str:
    """Format web search results into a prompt-ready context section.

    Parameters
    ----------
    results:
        List of search result dicts from :func:`search_web`.
    turn:
        1-indexed turn number used to tag the block and assemble
        turn-scoped citation tokens like ``[Tk:W1]``, ``[Tk:W2]``,
        .... Default ``1``.

    Returns
    -------
    A formatted string starting with ``## WEB SEARCH CONTEXT (turn k)``,
    or an empty string if no results are provided.
    """
    if not results:
        return ""

    lines: list[str] = [
        f"## WEB SEARCH CONTEXT (turn {turn})",
        "",
        "The following web search results provide real-world context "
        "for the research topic.  Each `model_summary` is a short "
        "model-written summary of the source (verbatim for the "
        "Anthropic backend, paraphrased for OpenAI / Gemini / Grok / "
        "DDG).  Use this information to ground your analysis in "
        "current facts (e.g., trade volumes, destinations, policies), "
        "but always apply the metacoupling framework rigorously.",
        "",
        "For subnational trade topics, some results may come from "
        "broader country-level searches when place-specific trade-partner "
        "data are sparse.  Treat those as proxy context and label them "
        "clearly rather than presenting them as direct subnational facts.",
        "",
        f"When you use a fact from a web result, cite it inline as "
        f"[T{turn}:W1], [T{turn}:W2], etc.  These are turn-scoped — the "
        f"`T{turn}:` prefix marks them as belonging to turn {turn} and "
        f"distinguishes them from literature citations [T{turn}:1], "
        f"[T{turn}:2], … that may be added separately.",
        "",
    ]

    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        model_summary = r.get("model_summary", "")
        url = r.get("url", "")
        lines.append(f"[T{turn}:W{i}] **{title}**")
        if model_summary:
            lines.append(f"   {model_summary}")
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


# JSON schema for OpenAI ``response_format`` strict mode on the
# structured map-extraction call inside ``extract_web_map_signals``.
# Mirrors the shape the function's normaliser expects (see the
# normalisation logic below).  Optional fields (``reason``,
# ``description``) are typed as strings rather than required because
# upstream parsers tolerate missing values, but they MUST be in the
# ``required`` list when ``strict=True`` is in effect — strict mode
# requires every property to be in ``required``.  Non-OpenAI clients
# never see this schema; they continue to use the prompt-based JSON
# path with ``_extract_json_object`` as the parser.
_WEB_MAP_SIGNALS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "focal_country": {"type": ["string", "null"]},
        "receiving_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "country": {"type": "string"},
                    "kind": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "country", "kind", "confidence",
                    "evidence", "reason",
                ],
            },
        },
        "spillover_systems": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "country": {"type": "string"},
                    "kind": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "country", "kind", "confidence",
                    "evidence", "reason",
                ],
            },
        },
        "flows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "source_country": {"type": "string"},
                    "target_country": {"type": "string"},
                    "kind": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                },
                "required": [
                    "category", "source_country", "target_country",
                    "kind", "confidence", "evidence", "description",
                ],
            },
        },
        "evidence_cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_id": {"type": "string"},
                    "claims_supported": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "relevance_score": {"type": "number"},
                    "source_type": {"type": "string"},
                },
                "required": [
                    "source_id", "claims_supported",
                    "relevance_score", "source_type",
                ],
            },
        },
        "suggested_followup_queries": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "focal_country", "receiving_systems", "spillover_systems",
        "flows", "evidence_cards", "suggested_followup_queries",
    ],
}


# PR #28: Anthropic submit-tool wrapping _WEB_MAP_SIGNALS_SCHEMA.
# Used by extract_web_map_signals when the llm_client is an
# AnthropicAdapter -- Claude returns the structured signals via a
# tool_use block (input matches the schema) instead of the strict
# json_schema text output that OpenAI provides.  Same canonical
# schema is wrapped in both code paths so the downstream
# normaliser sees identical structure regardless of provider.
_ANTHROPIC_WEB_MAP_SIGNALS_SUBMIT_TOOL: dict[str, object] = {
    "name": "submit_web_map_signals",
    "description": (
        "Submit the conservative, map-ready metacoupling signals "
        "extracted from the web search results.  Call this tool "
        "exactly once as the final action of your turn.  All "
        "fields are required; use null or empty lists when the "
        "results don't support a value."
    ),
    "input_schema": _WEB_MAP_SIGNALS_SCHEMA,
    "strict": True,
}


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
    """Use an LLM to extract validated map-ready signals from web results.

    Returns a dict containing ``focal_country``, ``receiving_systems``,
    ``spillover_systems``, and ``flows`` when extraction succeeds.
    """
    if not results:
        return None

    from metacouplingllm.knowledge.countries import (
        expand_supranational,
        get_country_name,
        resolve_country_code,
        supranational_display_name,
    )
    from metacouplingllm.llm.client import Message

    available_ids = {f"W{i}" for i in range(1, len(results) + 1)}
    result_lines: list[str] = []
    for idx, result in enumerate(results, 1):
        result_lines.append(f"[W{idx}] {result.get('title', '').strip()}")
        # Accept legacy "snippet" key from upstream providers as well.
        summary = (
            result.get("model_summary")
            or result.get("snippet", "")
        ).strip()
        result_lines.append(f"Summary: {summary}")
        result_lines.append(f"URL: {result.get('url', '').strip()}")
        result_lines.append("")

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
        "evidence_cards": [
            {
                "source_id": "W1",
                "claims_supported": [
                    "1-4 short factual claims this source can support",
                ],
                "relevance_score": 0.0,
                "source_type": (
                    "academic | government | international_organization | "
                    "ngo | news | industry | company | database | other"
                ),
            }
        ],
        "suggested_followup_queries": [
            "3-5 specific WEB search queries that would fill gaps "
            "in the current evidence",
        ],
    }
    system_text = (
        "You extract conservative, map-ready metacoupling signals from web "
        "search results.  Use only the provided results.  Do not invent "
        "countries, destinations, or flows. Return JSON only."
    )
    user_text = (
        f"Research query:\n{query.strip()}\n\n"
        "Extract map-ready countries and flows for a metacoupling map.\n"
        "\n"
        "Grounding rules (apply to EVERY field below):\n"
        "- Use only information from the provided web results.\n"
        "- Do NOT invent country names, ISO codes, or flows that "
        "are not supported by at least one source.\n"
        "- Do NOT claim a source supports a flow unless the source "
        "actually mentions both endpoints and the flow direction.\n"
        "- Do NOT extrapolate from one source to another -- each "
        "evidence_cards entry must cite its own W-id.\n"
        "\n"
        "Rules:\n"
        "- Use null or empty lists when uncertain.\n"
        "- Label items as 'proxy' when a result is broader than the focal "
        "study but still useful context.\n"
        "- Only include countries that are explicitly supported by the "
        "results.\n"
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
        "- Supranational unions are accepted: when a source names a "
        "union as a destination block (e.g., 'avocados are exported to "
        "the European Union', 'shipments to ASEAN markets') without "
        "naming individual member states, use the union name as the "
        "'country' value: 'European Union' (or 'EU'), 'ASEAN', 'USMCA', "
        "or 'NAFTA'.  The downstream map renderer treats these as "
        "single regions drawn at the union centroid with all members "
        "highlighted.  If the same source ALSO names specific member "
        "states (e.g., 'EU, Germany, France'), prefer the specific "
        "member states and omit the umbrella.\n"
        "- Most flows go FROM the focal country TO receiving countries "
        "(matter, information, energy, people, organisms).\n"
        "- Capital/payment flows go in REVERSE: FROM receiving countries "
        "TO the focal country.\n"
        "- Do NOT create flows from spillover/competitor countries to "
        "anyone. For example, if USA exports corn to Mexico "
        "and Brazil is a competitor, do NOT add a Brazil \u2192 Mexico flow.\n"
        "- Keep at most 6 receiving systems, 6 spillover systems, and 8 flows.\n"
        "- Every item must include evidence ids like W1.\n"
        "\n"
        "Evidence card rules:\n"
        "- Emit ONE evidence_cards entry per web result you saw, even when "
        "the result did not yield map signals.  Cards are independent of "
        "flows/systems -- they describe the web sources themselves.\n"
        "- source_id MUST match the result's W-id exactly (W1..W{n}).  Do "
        "not invent W-ids beyond the results shown.\n"
        "- claims_supported: 1-4 short factual phrases the source can "
        "genuinely support.  Do NOT paraphrase the source's summary "
        "wholesale -- list the SPECIFIC factual claims the source "
        "establishes.\n"
        "- relevance_score: a float in [0.0, 1.0] for the source's "
        "relevance to the research query.  Scores below 0.3 are dropped "
        "downstream.\n"
        "- source_type: pick from the enum.  When unsure, use 'other'.\n"
        "\n"
        "Suggested follow-up queries:\n"
        "- Emit 3-5 short, specific web-search query strings (each "
        "between 4 and 12 words) that would fill GAPS in the current "
        "web evidence.  Examples of good gap-filling queries: more recent "
        "data, missing subtopics, alternate framings, missing geographic "
        "regions, missing actor classes.\n"
        "- Do NOT repeat the research query verbatim.\n"
        "- These queries are WEB-specific suggestions; the main analysis "
        "downstream may also draw on local literature, so 'gap' here means "
        "a gap in WEB evidence only.\n\n"
        f"JSON schema example:\n{json.dumps(schema, ensure_ascii=True)}\n\n"
        "Web search results:\n"
        f"{chr(10).join(result_lines)}"
    )

    # Pass strict-output dispatch only for adapters that support it.
    # - OpenAIAdapter: response_format=json_schema (PR #17/#21).
    # - AnthropicAdapter: submit_web_map_signals tool + tool_choice
    #   forcing (PR #28).  Claude returns the structured signals via
    #   a tool_use block instead of JSON text.
    # - GeminiAdapter: native response_schema + response_mime_type
    #   combo (PR #30).  Gemini 3.x supports schema-constrained JSON
    #   output natively in one call.
    # - GrokAdapter: response_format=json_schema (PR #30).  Same
    #   shape as OpenAI's strict mode; xAI documents strict
    #   structured output at docs.x.ai/docs/guides/structured-outputs.
    # Other adapters fall through to the prompt-based JSON path;
    # ``_extract_json_object`` recovers the JSON from whatever
    # text they emit.  The fallback path also runs unconditionally
    # after the strict path for refusal / malformed-output resilience.
    chat_kwargs: dict[str, object] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    is_anthropic_dispatch = False
    try:
        from metacouplingllm.llm.client import OpenAIAdapter as _OpenAIAdapter
        from metacouplingllm.llm.client import (
            AnthropicAdapter as _AnthropicAdapter,
        )
        from metacouplingllm.llm.client import GeminiAdapter as _GeminiAdapter
        from metacouplingllm.llm.client import GrokAdapter as _GrokAdapter
        if isinstance(llm_client, _OpenAIAdapter):
            chat_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "web_map_signals",
                    "strict": True,
                    "schema": _WEB_MAP_SIGNALS_SCHEMA,
                },
            }
        elif isinstance(llm_client, _AnthropicAdapter):
            chat_kwargs["tools"] = [
                _ANTHROPIC_WEB_MAP_SIGNALS_SUBMIT_TOOL,
            ]
            chat_kwargs["tool_choice"] = {
                "type": "tool",
                "name": "submit_web_map_signals",
            }
            is_anthropic_dispatch = True
        elif isinstance(llm_client, _GeminiAdapter):
            # PR #30: Gemini's native strict-output combo.  Schema
            # is verified compatible with Gemini's response_schema
            # subset (no oneOf/anyOf, uses
            # ``{"type": ["string", "null"]}`` nullable form which
            # Gemini accepts).
            chat_kwargs["response_schema"] = _WEB_MAP_SIGNALS_SCHEMA
            chat_kwargs["response_mime_type"] = "application/json"
        elif isinstance(llm_client, _GrokAdapter):
            # PR #30: Grok's strict json_schema mode (same shape as
            # OpenAI's response_format above).
            chat_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "web_map_signals",
                    "strict": True,
                    "schema": _WEB_MAP_SIGNALS_SCHEMA,
                },
            }
    except Exception:
        # If the adapter import fails for any reason, fall through to
        # the prompt-based JSON path.
        pass

    response = llm_client.chat(
        messages=[
            Message(role="system", content=system_text),
            Message(role="user", content=user_text),
        ],
        **chat_kwargs,
    )

    # PR #28: Anthropic strict-output path -- Claude's structured
    # signals live in a ``submit_web_map_signals`` tool_use block,
    # not in the text content.  Try the tool path first; on failure
    # (no tool_use, or the tool_use input is malformed) fall through
    # to ``_extract_json_object`` so we still recover whatever JSON
    # Claude may have emitted as text.
    raw_obj: dict[str, object] | None = None
    if is_anthropic_dispatch and response.tool_uses:
        for tool_use in response.tool_uses:
            if tool_use.get("name") != "submit_web_map_signals":
                continue
            tool_input = tool_use.get("input")
            if isinstance(tool_input, dict):
                raw_obj = tool_input
                break
    if raw_obj is None:
        raw_obj = _extract_json_object(response.content)
    if raw_obj is None:
        return None

    def _resolve_country_or_supranational(
        raw_name: str,
    ) -> tuple[str | None, list[str] | None]:
        """Try ISO resolution first; fall back to supranational lookup.

        Returns ``(canonical_value, members)`` where:
        - On ISO success: ``(iso_alpha3_code, None)``
        - On supranational success: ``(display_name, [member_codes])``
          (e.g., ``("European Union", ["AUT", "BEL", ...])``)
        - On total failure: ``(None, None)``

        PR #22 taught the Stage-1 LLM to emit supranational unions
        (European Union / ASEAN / USMCA / NAFTA) as valid ``country``
        values; without this fallback the post-extraction validator
        silently drops every such entry.
        """
        if not raw_name:
            return (None, None)
        iso = resolve_country_code(raw_name)
        if iso:
            return (iso, None)
        members = expand_supranational(raw_name)
        if members:
            # Use the canonical display name regardless of how the
            # LLM spelled it ("EU" / "e.u." / "european union" all
            # collapse to "European Union").
            display = supranational_display_name(members) or raw_name
            return (display, list(members))
        return (None, None)

    def _normalise_country_entry(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        code, members = _resolve_country_or_supranational(
            str(item.get("country", "")).strip(),
        )
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
        result: dict[str, object] = {
            "country": code,
            "kind": kind,
            "confidence": confidence,
            "evidence": evidence,
            "reason": reason,
        }
        if members:
            result["supranational_members"] = members
        return result

    def _normalise_flow_entry(item: object) -> dict[str, object] | None:
        if not isinstance(item, dict):
            return None
        src_code, src_members = _resolve_country_or_supranational(
            str(item.get("source_country", "")).strip(),
        )
        tgt_code, tgt_members = _resolve_country_or_supranational(
            str(item.get("target_country", "")).strip(),
        )
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
        # For supranationals, ``code`` IS the display name already
        # (e.g., "European Union"); ``get_country_name`` returns
        # None and we fall through to the raw string.  For ISO
        # codes, ``get_country_name`` returns the country name.
        src_name = get_country_name(src_code) or src_code
        tgt_name = get_country_name(tgt_code) or tgt_code
        result: dict[str, object] = {
            "category": category,
            "source_country": src_code,
            "target_country": tgt_code,
            "direction": f"{src_name} \u2192 {tgt_name}",
            "description": description,
            "kind": kind,
            "confidence": confidence,
            "evidence": evidence,
        }
        if src_members:
            result["source_supranational_members"] = src_members
        if tgt_members:
            result["target_supranational_members"] = tgt_members
        return result

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

    # Evidence cards — per-result structured metadata.  Cards survive
    # even when no map signals were extracted (so users can still see
    # which sources were found, what they support, and how relevant
    # the model rated each).
    _ALLOWED_SOURCE_TYPES = frozenset({
        "academic", "government", "international_organization", "ngo",
        "news", "industry", "company", "database", "other",
    })
    evidence_cards: list[dict[str, object]] = []
    seen_card_ids: set[str] = set()
    raw_cards = raw_obj.get("evidence_cards", [])
    if isinstance(raw_cards, list):
        for item in raw_cards:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id", "")).strip().upper()
            if source_id not in available_ids or source_id in seen_card_ids:
                continue
            claims_raw = item.get("claims_supported", [])
            claims = (
                [str(c).strip() for c in claims_raw if str(c).strip()]
                if isinstance(claims_raw, list)
                else []
            )
            score = _coerce_confidence(item.get("relevance_score", 0.0))
            stype = str(item.get("source_type", "other")).strip().lower()
            if stype not in _ALLOWED_SOURCE_TYPES:
                stype = "other"
            seen_card_ids.add(source_id)
            evidence_cards.append({
                "source_id": source_id,
                "claims_supported": claims[:4],
                "relevance_score": score,
                "source_type": stype,
            })
    # Sort by W-id order (W1, W2, ...) for stable downstream rendering.
    evidence_cards.sort(key=lambda c: int(str(c["source_id"])[1:] or 0))

    # Suggested follow-up queries — short web-search strings.
    followup_queries: list[str] = []
    raw_followups = raw_obj.get("suggested_followup_queries", [])
    if isinstance(raw_followups, list):
        seen_followups: set[str] = set()
        for q in raw_followups:
            q_str = str(q).strip()
            if not q_str or q_str.lower() in seen_followups:
                continue
            # Cap each query at a sensible length to keep the prompt tidy.
            if len(q_str) > 160:
                q_str = q_str[:157].rstrip() + "..."
            seen_followups.add(q_str.lower())
            followup_queries.append(q_str)
            if len(followup_queries) >= 5:
                break

    if not any((
        focal_country, receiving, spillover, flows,
        evidence_cards, followup_queries,
    )):
        return None

    return {
        "focal_country": focal_country,
        "receiving_systems": receiving,
        "spillover_systems": spillover,
        "flows": flows,
        "evidence_cards": evidence_cards,
        "suggested_followup_queries": followup_queries,
    }


def format_web_map_signals_context(signals: dict[str, object] | None) -> str:
    """Format structured web map signals into a prompt-ready section."""
    if not signals:
        return ""

    lines = [
        "## STRUCTURED WEB MAP SIGNALS",
        "",
        "The following countries and flows were conservatively extracted "
        "from the web results and validated for mapping. Treat these as "
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

    # Evidence cards — per-result structured metadata.  Helps the main
    # analysis LLM weight sources by type and relevance, and quote
    # specific claims rather than paraphrasing.
    evidence_cards = signals.get("evidence_cards", [])
    if isinstance(evidence_cards, list) and evidence_cards:
        lines.append("")
        lines.append("### EVIDENCE CARDS")
        for card in evidence_cards:
            if not isinstance(card, dict):
                continue
            claims = card.get("claims_supported", [])
            claims_str = (
                "; ".join(str(c) for c in claims)
                if isinstance(claims, list) and claims
                else "(no specific claims listed)"
            )
            lines.append(
                f"- [{card.get('source_id', '?')}] "
                f"({card.get('source_type', 'other')}, "
                f"relevance={card.get('relevance_score', 0):.2f}) "
                f"supports: {claims_str}"
            )

    # Suggested follow-up queries — gaps the web extraction identified.
    # The main analysis LLM may incorporate these into its own coverage
    # assessment, and they surface to the user via AnalysisResult.
    followups = signals.get("suggested_followup_queries", [])
    if isinstance(followups, list) and followups:
        lines.append("")
        lines.append("### SUGGESTED FOLLOW-UP WEB SEARCHES")
        for q in followups:
            lines.append(f"- {q}")

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
    turn: int = 1,
) -> str:
    """Add inline turn-scoped ``[Tk:Wn]`` citations to formatted
    analysis lines.

    Uses web-search results and emits ``[Tk:W1]``, ``[Tk:W2]``, …
    so web references are visually distinct from literature
    citations ``[Tk:N]`` while sharing the turn-scoped grammar.

    Parameters
    ----------
    formatted:
        The formatted analysis text.
    web_results:
        Web search result dicts (each with ``title``, ``model_summary``,
        ``url``).
    min_keyword_overlap:
        Minimum shared keywords for a citation to be added.
    min_overlap_ratio:
        Minimum fraction of the line's keywords that must appear in the
        web result's ``model_summary``.
    turn:
        1-indexed turn number used to render citations as ``[Tk:Wn]``.
        Default ``1``.

    Returns
    -------
    The text with ``[Tk:Wn]`` citations appended to matching lines.
    """
    if not web_results:
        return formatted

    # Build keyword sets for each web result (title + model_summary).
    result_keywords: list[set[str]] = []
    for r in web_results:
        combined = (
            r.get("title", "") + " " + r.get("model_summary", "")
        ).strip()
        result_keywords.append(_web_tokenise(combined))

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

        # Find matching web results
        citations: list[int] = []
        for idx, sk in enumerate(result_keywords):
            overlap = line_tokens & sk
            n_overlap = len(overlap)
            ratio = n_overlap / len(line_tokens) if line_tokens else 0
            if n_overlap >= min_keyword_overlap and ratio >= min_overlap_ratio:
                citations.append(idx + 1)

        if citations:
            # Avoid duplicates if the LLM already cited a matching
            # turn-scoped web token in the line.
            existing: set[int] = set()
            for m in re.finditer(r"\[T(\d+):W(\d+)\]", line):
                if int(m.group(1)) == turn:
                    existing.add(int(m.group(2)))
            new_cites = [c for c in citations if c not in existing]
            if new_cites:
                cite_str = " " + " ".join(
                    f"[T{turn}:W{n}]" for n in new_cites
                )
                annotated.append(f"{line}{cite_str}")
            else:
                annotated.append(line)
        else:
            annotated.append(line)

    return "\n".join(annotated)
