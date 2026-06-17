"""Built-in run tracing for :class:`MetacouplingAssistant`.

When tracing is enabled (the default), the assistant wraps its LLM client in a
:class:`_RecordingClient` proxy that captures every ``chat()`` call, and after
each ``analyze()`` / ``refine()`` run it writes a folder of human-readable
artifacts (``00_run_config.md`` … ``10_pipeline_metadata.md`` + ``README.md`` +
``map.png``) plus attaches a :class:`RunTrace` to the result.

This is the package-internal version of what ``scripts/trace_pipeline.py`` used
to do by subclassing each provider adapter; the script now reuses this module so
there is a single tracing implementation.

The generated ``runs/`` directory is gitignored. Tracing never raises into the
caller: any failure while writing artifacts is swallowed (logged) and the
analysis result is still returned.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # avoid importing client types at module load
    from metacouplingllm.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults / opt-out
# ---------------------------------------------------------------------------
# ``trace=None`` on the assistant resolves to this module-level default (ON).
# The test suite flips it off via an autouse fixture; an env var provides a
# second, CI-friendly kill switch.
_DEFAULT_TRACE: bool = True
_DISABLE_ENV: str = "METACOUPLINGLLM_DISABLE_TRACE"


def default_trace_enabled() -> bool:
    """Resolve the effective default for ``trace=None`` (module flag AND env)."""
    if os.environ.get(_DISABLE_ENV, "").strip().lower() in ("1", "true", "yes"):
        return False
    return _DEFAULT_TRACE


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class CallRecord:
    """One captured ``chat()`` call."""

    n: int
    label: str
    messages: list[dict[str, str]]
    requested_temperature: float
    max_tokens: int | None
    extra_kwargs: dict[str, Any]
    response_content: str
    response_tool_uses: list[dict[str, object]] | None
    usage: dict[str, int]
    duration_s: float


@dataclass
class RunTrace:
    """Everything captured for one ``analyze()`` / ``refine()`` run.

    Attached to the result as ``result.trace``. ``out_dir`` is the folder the
    artifacts were written to, or ``None`` if writing was disabled or failed.
    """

    query: str
    model: str
    calls: list[CallRecord]
    wall_clock_s: float
    git_sha: str
    git_branch: str
    env: dict[str, str]
    total_input_tokens: int
    total_output_tokens: int
    out_dir: Path | None = None

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


# ---------------------------------------------------------------------------
# Recording proxy
# ---------------------------------------------------------------------------
class _RecordingClient:
    """Wraps any :class:`LLMClient`, recording every ``chat()`` call.

    Forwards ``chat()`` (and, via ``__getattr__``, any other attribute) to the
    wrapped client unchanged, so it is a drop-in stand-in EXCEPT for
    ``isinstance`` checks -- callers that branch on the concrete adapter type
    (e.g. web-search auto-wiring) must use the unwrapped inner client.
    """

    def __init__(self, inner: "LLMClient") -> None:
        self._inner = inner
        self.captured_calls: list[CallRecord] = []
        self._current_label = "unlabeled"

    @contextmanager
    def label(self, name: str) -> Iterator[None]:
        prev = self._current_label
        self._current_label = name
        try:
            yield
        finally:
            self._current_label = prev

    def chat(
        self,
        messages: list[Any],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Any:
        t0 = time.perf_counter()
        response = self._inner.chat(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )
        duration = time.perf_counter() - t0
        self.captured_calls.append(
            CallRecord(
                n=len(self.captured_calls) + 1,
                label=self._current_label,
                messages=[
                    {"role": getattr(m, "role", "?"),
                     "content": getattr(m, "content", "")}
                    for m in messages
                ],
                requested_temperature=temperature,
                max_tokens=max_tokens,
                extra_kwargs=dict(kwargs),
                response_content=getattr(response, "content", "") or "",
                response_tool_uses=getattr(response, "tool_uses", None),
                usage=dict(getattr(response, "usage", {}) or {}),
                duration_s=round(duration, 2),
            )
        )
        return response

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal lookup fails; ``_inner`` is a real attribute
        # so it is never routed here (guard anyway against init-time recursion).
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Usage normalization (handles both provider schemas)
# ---------------------------------------------------------------------------
def _norm_usage(usage: dict[str, int]) -> tuple[int, int, int]:
    """Return (input, output, total) tokens from either usage schema.

    OpenAI/Grok use prompt_tokens/completion_tokens/total_tokens; Anthropic and
    Gemini use input_tokens/output_tokens.
    """
    inp = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    out = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    tot = int(usage.get("total_tokens", inp + out) or (inp + out))
    return inp, out, tot


def _sum_usage(calls: list[CallRecord]) -> tuple[int, int]:
    total_in = total_out = 0
    for c in calls:
        i, o, _ = _norm_usage(c.usage)
        total_in += i
        total_out += o
    return total_in, total_out


# ---------------------------------------------------------------------------
# Environment / git capture
# ---------------------------------------------------------------------------
def _git_info() -> tuple[str, str]:
    sha = branch = "unknown"
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass
    return sha, branch


def _env_info() -> dict[str, str]:
    env: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("metacouplingllm", "numpy", "fastembed",
                "matplotlib", "geopandas", "openai", "anthropic"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "<unknown>")
        except Exception as exc:
            env[mod] = f"<not installed: {exc}>"
    return env


# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------
def slug_dir(base: Path, query: str) -> Path:
    """Return ``base / <utc-timestamp>_<query-slug>`` (session root)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "_", (query or "run").lower()).strip("_")[:40]
    return base / f"{stamp}_{slug or 'run'}"


# ---------------------------------------------------------------------------
# Trace assembly
# ---------------------------------------------------------------------------
def build_run_trace(
    proxy: _RecordingClient,
    *,
    query: str,
    model: str,
    wall_clock_s: float,
) -> RunTrace:
    calls = list(proxy.captured_calls)
    total_in, total_out = _sum_usage(calls)
    sha, branch = _git_info()
    return RunTrace(
        query=query,
        model=model,
        calls=calls,
        wall_clock_s=round(wall_clock_s, 2),
        git_sha=sha,
        git_branch=branch,
        env=_env_info(),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
    )


# ---------------------------------------------------------------------------
# Markdown helpers (moved from scripts/trace_pipeline.py)
# ---------------------------------------------------------------------------
def _h2(s: str) -> str:
    return f"## {s}\n\n"


def _fence_for(text: str) -> str:
    fence = "````"
    while fence in (text or ""):
        fence += "`"
    return fence


def _kv_table(d: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in d.items():
        v_str = str(v).replace("|", "\\|").replace("\n", " ")
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        lines.append(f"| `{k}` | {v_str} |")
    return "\n".join(lines) + "\n"


def _json_block(obj: Any) -> str:
    def default(o: Any):
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    try:
        text = json.dumps(obj, indent=2, default=default, ensure_ascii=False)
    except Exception as exc:
        text = f"<un-serializable object: {exc}>\n{obj!r}"
    return f"```json\n{text}\n```\n"


def _llm_call_md(call: CallRecord, title: str, context: str) -> str:
    inp, out, tot = _norm_usage(call.usage)
    parts = [f"# {title}\n\n", context, "\n\n", _h2("Call metadata")]
    parts.append(_kv_table({
        "call_index": call.n,
        "label": call.label,
        "requested_temperature": call.requested_temperature,
        "max_tokens": call.max_tokens,
        "duration_s": call.duration_s,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": tot,
        "extra_kwargs": call.extra_kwargs or "{}",
    }))
    parts.append("\n")
    for i, msg in enumerate(call.messages):
        content = msg.get("content", "")
        parts.append(_h2(
            f"Message {i + 1} — role: `{msg.get('role', '?')}` "
            f"({len(content)} chars)"
        ))
        fence = _fence_for(content)
        parts.append(f"{fence}\n{content}\n{fence}\n\n")
    parts.append(_h2(f"Response ({len(call.response_content)} chars)"))
    fence = _fence_for(call.response_content)
    parts.append(f"{fence}\n{call.response_content}\n{fence}\n")
    if call.response_tool_uses:
        parts.append("\n" + _h2("Tool uses"))
        parts.append(_json_block(call.response_tool_uses))
    return "".join(parts)


def _call_by_label(trace: RunTrace, label: str) -> CallRecord | None:
    return next((c for c in trace.calls if c.label == label), None)


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------
def write_run_artifacts(
    out_dir: Path,
    *,
    trace: RunTrace,
    assistant: Any,
    result: Any,
) -> list[Path]:
    """Write the 00–10 + README + map artifacts. Never raises; returns the
    list of files written (empty on failure)."""
    written: list[Path] = []

    def _write(name: str, content: str) -> None:
        p = out_dir / name
        p.write_text(content, encoding="utf-8")
        written.append(p)

    def _attr(name: str, default: Any = None) -> Any:
        return getattr(assistant, name, default)

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- 00 run_config -------------------------------------------------
        cfg = "# 00 — Run configuration\n\n"
        cfg += _h2("Query") + f"> {trace.query}\n\n"
        cfg += _h2("Timestamps") + _kv_table({
            "utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_s": trace.wall_clock_s,
        })
        cfg += "\n" + _h2("Git") + _kv_table(
            {"branch": trace.git_branch, "sha": trace.git_sha})
        cfg += "\n" + _h2("Model") + _kv_table({"model": trace.model})
        cfg += "\n" + _h2("Assistant parameters") + _kv_table({
            "rag_top_k": _attr("_rag_top_k"),
            "rag_max_chunks_per_paper": _attr("_rag_max_chunks_per_paper"),
            "rag_backend": _attr("_rag_backend"),
            "rag_structured_extraction": _attr("_rag_structured_extraction"),
            "web_search": _attr("_web_search"),
            "web_search_max_results": _attr("_web_search_max_results"),
            "web_structured_extraction": _attr("_web_structured_extraction"),
            "auto_map": _attr("_auto_map"),
            "temperature": _attr("_temperature"),
            "coupling_analysis": _attr("_coupling_analysis"),
            "verbose": _attr("_verbose"),
        })
        cfg += "\n" + _h2("Environment") + _kv_table(trace.env)
        _write("00_run_config.md", cfg)

        web_results = _attr("_last_web_results") or []
        rag_only = getattr(result, "parsed", None) is None

        # --- 01 web_results_raw -------------------------------------------
        web_md = "# 01 — Raw web search results\n\n"
        web_md += f"`_last_web_results` — {len(web_results)} entries.\n\n"
        for i, hit in enumerate(web_results, start=1):
            web_md += f"### Result {i}\n\n"
            web_md += _kv_table({
                k: v for k, v in hit.items()
                if k not in {"model_summary", "snippet"}
            })
            summary = hit.get("model_summary") or hit.get("snippet", "")
            if summary:
                web_md += f"\n**Model summary:**\n\n> {summary}\n\n"
        _write("01_web_results_raw.md", web_md)

        # --- 02 web extraction call (NOT captured by the proxy: the web
        # backend calls the raw client directly — render from intermediates).
        web_call = _call_by_label(trace, "web_extraction")
        if web_call is not None:
            _write("02_llm_call_web_extraction.md", _llm_call_md(
                web_call, "02 — LLM call: structured web extraction",
                "Distills the raw web results (01) into structured map signals (03)."))
        else:
            _write(
                "02_llm_call_web_extraction.md",
                "# 02 — LLM call: structured web extraction\n\n"
                "_Not captured by the in-process trace: the web-extraction call "
                "is issued by the native web-search backend via the provider's "
                "raw client, not through the assistant's `chat()` proxy. The "
                "raw inputs/outputs are in files 01 and 03._\n",
            )

        # --- 03 web structured signals ------------------------------------
        sig_md = "# 03 — Web structured signals\n\n"
        sig_md += "Parsed map signals derived from the web results.\n\n"
        sig_md += _json_block(_attr("_last_web_map_signals"))
        _write("03_web_structured_signals.md", sig_md)

        # --- 04 rag_chunks -------------------------------------------------
        hits = _attr("_last_rag_hits") or []
        rag_md = "# 04 — Retrieved RAG chunks\n\n"
        rag_md += (
            f"`_last_rag_hits` — {len(hits)} chunks "
            f"(`rag_top_k={_attr('_rag_top_k')}`, "
            f"`rag_max_chunks_per_paper={_attr('_rag_max_chunks_per_paper')}`)\n\n"
        )
        for i, hit in enumerate(hits, start=1):
            chunk = getattr(hit, "chunk", None)
            if chunk is None:
                continue
            rag_md += (
                f"### Chunk {i} — `{getattr(chunk, 'paper_key', '?')}` "
                f"(score: {getattr(hit, 'score', 0.0):.4f})\n\n"
            )
            text = getattr(chunk, "text", "")
            rag_md += _kv_table({
                "paper_title": getattr(chunk, "paper_title", ""),
                "authors": getattr(chunk, "authors", ""),
                "year": getattr(chunk, "year", ""),
                "section": getattr(chunk, "section", ""),
                "chunk_index": getattr(chunk, "chunk_index", ""),
                "char_count": len(text),
            })
            fence = _fence_for(text)
            rag_md += f"\n**Full chunk text:**\n\n{fence}\n{text}\n{fence}\n\n"
        _write("04_rag_chunks.md", rag_md)

        # --- 05 main analysis / rag_qa call -------------------------------
        main_call = (_call_by_label(trace, "main_analysis")
                     or _call_by_label(trace, "rag_qa"))
        if main_call is not None:
            title = ("05 — LLM call: RAG-only Q&A" if main_call.label == "rag_qa"
                     else "05 — LLM call: main framework analysis")
            _write("05_llm_call_main_analysis.md", _llm_call_md(
                main_call, title,
                "The primary model call. The user message embeds the retrieved "
                "RAG chunks (04) and web results (01); the response carries the "
                "analysis with `[Tk:N]` / `[Tk:Wn]` citations."))

        # --- 06 parsed_analysis (framework mode only) ---------------------
        parsed = getattr(result, "parsed", None)
        if parsed is not None:
            pdict: dict[str, Any] = {}
            for attr in dir(parsed):
                if attr.startswith("_"):
                    continue
                try:
                    val = getattr(parsed, attr)
                except Exception:
                    continue
                if callable(val):
                    continue
                pdict[attr] = val
            _write("06_parsed_analysis.md",
                   "# 06 — Parsed analysis\n\n"
                   "`result.parsed` re-parsed into structured fields.\n\n"
                   + _json_block(pdict))

        # --- 07 map_extraction call + 08 map_data -------------------------
        map_call = _call_by_label(trace, "map_extraction")
        if map_call is not None:
            _write("07_llm_call_map_extraction.md", _llm_call_md(
                map_call, "07 — LLM call: structured map extraction",
                "Re-extracts the prose analysis into strict JSON the renderer "
                "consumes (output in 08)."))
        if parsed is not None:
            _write("08_map_data.md",
                   "# 08 — Map data\n\n`parsed.map_data` — structured input to "
                   "the renderer.\n\n"
                   + _json_block(getattr(parsed, "map_data", None)))

        # --- 09 formatted_output ------------------------------------------
        formatted = (getattr(result, "formatted", None)
                     or getattr(result, "answer", "") or "")
        fence = _fence_for(formatted)
        _write("09_formatted_output.md",
               "# 09 — Formatted output (what the user sees)\n\n"
               f"{fence}\n{formatted}\n{fence}\n")

        # --- 10 pipeline_metadata -----------------------------------------
        meta = "# 10 — Pipeline metadata\n\n"
        if not rag_only:
            meta += _h2("Map render") + _kv_table({
                "last_map_type": _attr("_last_map_type"),
                "last_map_notice": _attr("_last_map_notice"),
                "result.map_present": getattr(result, "map", None) is not None,
            }) + "\n"
        meta += _h2("Token usage (per LLM call)")
        meta += "| Call | Label | Input | Output | Total | Duration (s) |\n"
        meta += "|---|---|---|---|---|---|\n"
        for c in trace.calls:
            inp, out, tot = _norm_usage(c.usage)
            meta += (f"| {c.n} | {c.label} | {inp} | {out} | {tot} "
                     f"| {c.duration_s} |\n")
        meta += (f"| **TOTAL** | — | **{trace.total_input_tokens}** "
                 f"| **{trace.total_output_tokens}** "
                 f"| **{trace.total_tokens}** | — |\n")
        llm_total = round(sum(c.duration_s for c in trace.calls), 2)
        meta += "\n" + _h2("Wall-clock breakdown") + _kv_table({
            "total_wall_clock_s": trace.wall_clock_s,
            "sum_of_llm_call_s": llm_total,
            "non_llm_s (retrieval + render + parsing)":
                round(trace.wall_clock_s - llm_total, 2),
        })
        _write("10_pipeline_metadata.md", meta)

        # --- map.png -------------------------------------------------------
        fig = getattr(result, "map", None)
        if fig is not None:
            try:
                fig.savefig(out_dir / "map.png", dpi=150, bbox_inches="tight")
                written.append(out_dir / "map.png")
            except Exception as exc:
                logger.warning("Trace: failed to save map figure: %s", exc)

        # --- README --------------------------------------------------------
        readme = "# Pipeline trace\n\n"
        readme += f"**Query:** {trace.query}\n\n"
        readme += f"**Model:** `{trace.model}`\n\n"
        readme += (f"**Wall-clock:** {trace.wall_clock_s:.1f}s | "
                   f"**LLM calls:** {len(trace.calls)} | "
                   f"**Total tokens:** {trace.total_tokens}\n\n")
        readme += _h2("Reading this trace")
        readme += (
            "- Files are numbered by **pipeline stage** (the order steps "
            "run), not by model-call order. For example, the structured "
            "web-extraction call appears early, at `02`, because it runs "
            "during the web-search stage, before the main analysis.\n"
            "- The **LLM calls** count above, and the token table in "
            "`10_pipeline_metadata.md`, include only calls captured through "
            "the assistant's `chat()` proxy. The structured web-extraction "
            "call (`02`, when present) is *summarized rather than "
            "chat-captured*, because it is issued by the provider's native "
            "web-search client rather than that proxy; its raw inputs and "
            "outputs are in `01` and `03`.\n"
            "- Not every file appears in every run: the web, RAG, map, and "
            "supplementary stages are written only when the corresponding "
            "feature is enabled.\n\n"
        )
        descriptions = {
            "00_run_config.md": "Query, model, parameters, git SHA, and environment.",
            "01_web_results_raw.md": "Raw results returned by the web search.",
            "02_llm_call_web_extraction.md": "Structured web-extraction model call (summarized; see above).",
            "03_web_structured_signals.md": "Structured signals and evidence cards from the web results.",
            "04_rag_chunks.md": "Literature passages retrieved from the RAG corpus.",
            "05_llm_call_main_analysis.md": "Main framework-analysis model call, with the full seven-layer system prompt.",
            "06_parsed_analysis.md": "Parsed ParsedAnalysis structure (parser output of the main response).",
            "07_llm_call_map_extraction.md": "Map-signal extraction model call.",
            "08_map_data.md": "Structured map data used to render the figure.",
            "09_formatted_output.md": "Final formatted report.",
            "10_pipeline_metadata.md": "Per-call token usage, wall-clock breakdown, and map metadata.",
            "map.png": "Rendered metacoupling map.",
            "README.md": "This file.",
        }
        readme += _h2("Artifacts")
        readme += "| File | Description |\n|---|---|\n"
        for fname in sorted(p.name for p in written):
            readme += f"| [`{fname}`](./{fname}) | {descriptions.get(fname, '')} |\n"
        _write("README.md", readme)

    except Exception as exc:  # never propagate into the caller
        logger.warning("Trace: failed to write artifacts to %s: %s",
                       out_dir, exc)
        return written

    return written
