#!/usr/bin/env python3
"""Trace MetacouplingAssistant.analyze() end-to-end and dump every
intermediate artifact to disk for study.

Wraps the OpenAI client so every ``chat()`` call is logged with full
request + response + usage + duration. After the run, dumps:

  00_run_config.md            Config + env + git SHA
  01_web_results_raw.md       Raw web search results
  02_llm_call_1_web_extraction.md   LLM call #1 (structured web extraction)
  03_web_structured_signals.md      Parsed output of call #1
  04_rag_chunks.md            Retrieved RAG chunks (full text)
  05_llm_call_2_main_analysis.md    LLM call #2 (main framework analysis)
  06_parsed_analysis.md       Parsed structured fields
  07_llm_call_3_map_extraction.md   LLM call #3 (structured map extraction)
  08_map_data.md              Parsed output of call #3
  09_formatted_output.md      Human-readable final text
  10_pipeline_metadata.md     Map type, notices, warnings, token usage, timing
  README.md                   Index of all artifacts
  map.png                     Rendered matplotlib figure (if auto_map produced one)

Usage::

    python scripts/trace_pipeline.py

Edit ``QUERY``, ``OUT_DIR``, ``MODEL`` at the top to retarget.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

QUERY = "Impact of avocado production and trade in Mexico on sustainability"
OUT_DIR = Path("runs/avocado_2026-05-18")
MODEL = "gpt-5.5"
API_KEY_PATH = Path(
    r"D:\Onedrive\OneDrive - Michigan State University\Desktop\api.env"
)


# ---------------------------------------------------------------------------
# Load API key from .env file (no python-dotenv dependency)
# ---------------------------------------------------------------------------

def load_api_key() -> None:
    """Read OPENAI_API_KEY from the configured .env file and inject into env."""
    if not API_KEY_PATH.exists():
        sys.exit(f"ERROR: API key file not found at {API_KEY_PATH}")
    for raw in API_KEY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY not found in api.env")


# ---------------------------------------------------------------------------
# Logging adapter — captures every LLM chat() call
# ---------------------------------------------------------------------------

def build_logging_adapter():
    """Return a LoggingOpenAIAdapter class bound to the project's OpenAIAdapter."""
    from metacouplingllm.llm.client import OpenAIAdapter, LLMResponse, Message

    class LoggingOpenAIAdapter(OpenAIAdapter):
        """OpenAIAdapter that records every chat() call for later dumping."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.captured_calls: list[dict[str, Any]] = []

        def chat(
            self,
            messages: list[Message],
            temperature: float = 0.7,
            max_tokens: int | None = None,
            **kwargs: Any,
        ) -> LLMResponse:
            # gpt-5-family models only accept their default temperature
            # (=1) and require max_completion_tokens instead of max_tokens.
            # The adapter retries max_tokens automatically, but the
            # internal structured-extraction calls hardcode temperature=0.0,
            # so we coerce here to avoid the 400 + skip-the-call path.
            #
            # ``**kwargs`` forwards forward-compat kwargs the library
            # adds over time (e.g. ``response_format`` for strict
            # json_schema mode).  Without this passthrough, a wrapper
            # that subclasses ``OpenAIAdapter`` with a closed signature
            # would crash whenever the library introduced a new chat()
            # kwarg.
            effective_temp = temperature
            if self._model.startswith("gpt-5"):
                effective_temp = 1.0
            t0 = time.perf_counter()
            response = super().chat(
                messages,
                temperature=effective_temp,
                max_tokens=max_tokens,
                **kwargs,
            )
            duration = time.perf_counter() - t0
            self.captured_calls.append({
                "n": len(self.captured_calls) + 1,
                "messages": [
                    {"role": m.role, "content": m.content} for m in messages
                ],
                "requested_temperature": temperature,
                "effective_temperature": effective_temp,
                "max_tokens": max_tokens,
                # Record forward-compat kwargs (e.g. response_format)
                # so artifacts show which calls were made in strict
                # json_schema mode vs prompt-based JSON.
                "extra_kwargs": dict(kwargs),
                "response_content": response.content,
                "usage": dict(response.usage) if response.usage else {},
                "duration_s": round(duration, 2),
            })
            return response

    return LoggingOpenAIAdapter


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def _h2(s: str) -> str:
    return f"## {s}\n\n"


def _kv_table(d: dict[str, Any]) -> str:
    """Render a dict as a Markdown table."""
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in d.items():
        v_str = str(v).replace("|", "\\|").replace("\n", " ")
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        lines.append(f"| `{k}` | {v_str} |")
    return "\n".join(lines) + "\n"


def _json_block(obj: Any) -> str:
    """Render any object as a fenced JSON code block (best-effort)."""
    def default(o: Any):
        if is_dataclass(o):
            return asdict(o)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    try:
        text = json.dumps(obj, indent=2, default=default, ensure_ascii=False)
    except Exception as exc:
        text = f"<un-serializable object: {exc}>\n{obj!r}"
    return f"```json\n{text}\n```\n"


def _llm_call_md(call: dict[str, Any], title: str, context: str) -> str:
    """Render a captured LLM call as a structured markdown document."""
    out = [f"# {title}\n\n", context, "\n\n"]
    out.append(_h2("Call metadata"))
    out.append(_kv_table({
        "call_index": call["n"],
        "requested_temperature": call["requested_temperature"],
        "effective_temperature": call["effective_temperature"],
        "max_tokens": call["max_tokens"],
        "duration_s": call["duration_s"],
        "input_tokens": call["usage"].get("prompt_tokens", "?"),
        "output_tokens": call["usage"].get("completion_tokens", "?"),
        "total_tokens": call["usage"].get("total_tokens", "?"),
    }))
    out.append("\n")
    for i, msg in enumerate(call["messages"]):
        out.append(_h2(f"Message {i+1} — role: `{msg['role']}` ({len(msg['content'])} chars)"))
        # Use a fenced block; pick a fence that won't clash with content.
        fence = "````"
        while fence in msg["content"]:
            fence += "`"
        out.append(f"{fence}\n{msg['content']}\n{fence}\n\n")
    out.append(_h2(f"Response ({len(call['response_content'])} chars)"))
    fence = "````"
    while fence in call["response_content"]:
        fence += "`"
    out.append(f"{fence}\n{call['response_content']}\n{fence}\n")
    return "".join(out)


def write_artifacts(
    out_dir: Path,
    *,
    query: str,
    model: str,
    assistant,
    result,
    captured_calls: list[dict[str, Any]],
    wall_clock_s: float,
) -> list[Path]:
    """Write every artifact MD file and the map PNG to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- 00 run_config -----------------------------------------------------
    git_sha = "unknown"
    git_branch = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True,
        ).strip()
    except Exception:
        pass

    versions = {}
    for mod in ("metacouplingllm", "numpy", "fastembed",
                "matplotlib", "geopandas", "openai"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "<unknown>")
        except Exception as exc:
            versions[mod] = f"<not installed: {exc}>"

    config_md = "# 00 — Run configuration\n\n"
    config_md += _h2("Query")
    config_md += f"> {query}\n\n"
    config_md += _h2("Timestamps")
    config_md += _kv_table({
        "utc": datetime.now(timezone.utc).isoformat(),
        "local": datetime.now().isoformat(),
        "wall_clock_s": round(wall_clock_s, 2),
    })
    config_md += "\n" + _h2("Git")
    config_md += _kv_table({"branch": git_branch, "sha": git_sha})
    config_md += "\n" + _h2("Model / API")
    config_md += _kv_table({
        "model": model,
        "api_key_source": str(API_KEY_PATH),
        "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
    })
    config_md += "\n" + _h2("Assistant parameters")
    config_md += _kv_table({
        "rag_top_k": assistant._rag_top_k,
        "rag_max_chunks_per_paper": assistant._rag_max_chunks_per_paper,
        "rag_backend": assistant._rag_backend,
        "rag_mode": assistant._rag_mode,
        "rag_structured_extraction": assistant._rag_structured_extraction,
        "web_search": assistant._web_search,
        "web_search_max_results": assistant._web_search_max_results,
        "web_structured_extraction": assistant._web_structured_extraction,
        "auto_map": assistant._auto_map,
        "temperature": assistant._temperature,
        "coupling_analysis": assistant._coupling_analysis,
        "verbose": assistant._verbose,
    })
    config_md += "\n" + _h2("Environment")
    config_md += _kv_table({
        "python": platform.python_version(),
        "platform": platform.platform(),
        **versions,
    })
    p = out_dir / "00_run_config.md"
    p.write_text(config_md, encoding="utf-8")
    written.append(p)

    # --- 01 web_results_raw ------------------------------------------------
    web_md = "# 01 — Raw web search results\n\n"
    web_md += f"`self._last_web_results` — {len(assistant._last_web_results)} entries.\n\n"
    for i, hit in enumerate(assistant._last_web_results, start=1):
        web_md += f"### Result {i}\n\n"
        # Hide the long-form summary from the metadata table; render it
        # as a quote block below.  Accept the legacy ``snippet`` key as
        # a fallback for older backends that haven't migrated yet.
        web_md += _kv_table({
            k: v
            for k, v in hit.items()
            if k not in {"model_summary", "snippet"}
        })
        model_summary = hit.get("model_summary") or hit.get("snippet", "")
        if model_summary:
            web_md += f"\n**Model summary:**\n\n> {model_summary}\n\n"
    p = out_dir / "01_web_results_raw.md"
    p.write_text(web_md, encoding="utf-8")
    written.append(p)

    # --- 02-03 web extraction LLM call + structured signals ----------------
    if captured_calls:
        web_call_md = _llm_call_md(
            captured_calls[0],
            title="02 — LLM call #1: structured web extraction",
            context=(
                "First LLM call in the pipeline. Distills the raw web "
                "search results above (file 01) into structured map "
                "signals (file 03). Triggered because `web_search=True` "
                "AND `auto_map=True` auto-enabled `web_structured_extraction`."
            ),
        )
        p = out_dir / "02_llm_call_1_web_extraction.md"
        p.write_text(web_call_md, encoding="utf-8")
        written.append(p)

    signals_md = "# 03 — Web structured signals\n\n"
    signals_md += (
        "Parsed output of LLM call #1 (file 02). This is what gets "
        "appended to the main analysis prompt as additional structured "
        "context AND feeds the map renderer.\n\n"
    )
    signals_md += _json_block(assistant._last_web_map_signals)
    p = out_dir / "03_web_structured_signals.md"
    p.write_text(signals_md, encoding="utf-8")
    written.append(p)

    # --- 04 rag_chunks -----------------------------------------------------
    rag_md = "# 04 — Retrieved RAG chunks\n\n"
    hits = assistant._last_rag_hits or []
    rag_md += f"`self._last_rag_hits` — {len(hits)} chunks "
    rag_md += f"(`rag_top_k={assistant._rag_top_k}`, "
    rag_md += f"`rag_max_chunks_per_paper={assistant._rag_max_chunks_per_paper}`)\n\n"
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        rag_md += f"### Chunk {i} — `{chunk.paper_key}` (score: {hit.score:.4f})\n\n"
        rag_md += _kv_table({
            "paper_title": chunk.paper_title,
            "authors": chunk.authors,
            "year": chunk.year,
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "char_count": len(chunk.text),
            "word_count": len(chunk.text.split()),
        })
        rag_md += "\n**Full chunk text:**\n\n"
        fence = "````"
        while fence in chunk.text:
            fence += "`"
        rag_md += f"{fence}\n{chunk.text}\n{fence}\n\n"
    p = out_dir / "04_rag_chunks.md"
    p.write_text(rag_md, encoding="utf-8")
    written.append(p)

    # --- 05 main_analysis call --------------------------------------------
    if len(captured_calls) >= 2:
        main_md = _llm_call_md(
            captured_calls[1],
            title="05 — LLM call #2: main framework analysis",
            context=(
                "Second LLM call. The system prompt teaches the LLM the "
                "Liu 2017 metacoupling framework and citation grammar; "
                "the user message embeds the retrieved RAG chunks (file "
                "04) inside a `<retrieved_literature>` XML block, the "
                "raw web results (file 01) inside a `<web_search_results>` "
                "block, the structured web signals (file 03), and the "
                "research question. The LLM responds with the analysis "
                "using `[T1:N]` literature citations and `[T1:Wn]` web "
                "citations."
            ),
        )
        p = out_dir / "05_llm_call_2_main_analysis.md"
        p.write_text(main_md, encoding="utf-8")
        written.append(p)

    # --- 06 parsed_analysis ------------------------------------------------
    parsed_md = "# 06 — Parsed analysis\n\n"
    parsed_md += (
        "`AnalysisResult.parsed` — the LLM response (file 05) re-parsed "
        "into structured fields. This is what downstream consumers "
        "(formatter, map extractor, pericoupling validators) read.\n\n"
    )
    parsed = result.parsed
    parsed_dict: dict[str, Any] = {}
    for attr in dir(parsed):
        if attr.startswith("_") or callable(getattr(parsed, attr)):
            continue
        try:
            parsed_dict[attr] = getattr(parsed, attr)
        except Exception:
            parsed_dict[attr] = "<error reading>"
    parsed_md += _json_block(parsed_dict)
    p = out_dir / "06_parsed_analysis.md"
    p.write_text(parsed_md, encoding="utf-8")
    written.append(p)

    # --- 07 map_extraction call -------------------------------------------
    if len(captured_calls) >= 3:
        map_call_md = _llm_call_md(
            captured_calls[2],
            title="07 — LLM call #3: structured map extraction",
            context=(
                "Third LLM call. Takes the prose analysis (file 05 "
                "response) and re-extracts it into a strict JSON schema "
                "the matplotlib renderer can consume: ISO-3 country "
                "codes, validated flow categories, parseable endpoint "
                "pairs. Output is in file 08."
            ),
        )
        p = out_dir / "07_llm_call_3_map_extraction.md"
        p.write_text(map_call_md, encoding="utf-8")
        written.append(p)

    # --- 08 map_data -------------------------------------------------------
    map_md = "# 08 — Map data (structured output of call #3)\n\n"
    map_md += (
        "`parsed.map_data` — the structured map signals the renderer "
        "consumes. Includes `focal_country`, `adm1_region`, "
        "`receiving_countries`, `spillover_countries`, and a `flows` "
        "list with canonical ISO-3 codes and Liu 2017 flow categories.\n\n"
    )
    map_md += _json_block(getattr(parsed, "map_data", None))
    p = out_dir / "08_map_data.md"
    p.write_text(map_md, encoding="utf-8")
    written.append(p)

    # --- 09 formatted_output ----------------------------------------------
    formatted_md = "# 09 — Formatted output (what the user sees)\n\n"
    formatted_md += "`AnalysisResult.formatted` — full human-readable text.\n\n"
    fence = "````"
    while fence in result.formatted:
        fence += "`"
    formatted_md += f"{fence}\n{result.formatted}\n{fence}\n"
    p = out_dir / "09_formatted_output.md"
    p.write_text(formatted_md, encoding="utf-8")
    written.append(p)

    # --- 10 pipeline_metadata ---------------------------------------------
    meta_md = "# 10 — Pipeline metadata\n\n"
    meta_md += _h2("Map render")
    meta_md += _kv_table({
        "last_map_type": assistant._last_map_type,
        "last_map_notice": assistant._last_map_notice,
        "result.map_notice": result.map_notice,
        "result.map_present": result.map is not None,
    })
    meta_md += "\n" + _h2("Flow parse warnings (legacy regex path)")
    warnings = assistant._last_flow_parse_warnings or []
    if warnings:
        meta_md += _json_block(warnings)
    else:
        meta_md += "_(none)_\n\n"
    meta_md += _h2("Citation accounting")
    meta_md += _kv_table({
        "turn_number": result.turn_number,
        "turn_passage_counts": dict(assistant._turn_passage_counts),
        "turn_web_counts": dict(assistant._turn_web_counts),
    })
    meta_md += "\n" + _h2("Token usage (per LLM call)")
    meta_md += "| Call | Purpose | Input | Output | Total | Duration (s) |\n"
    meta_md += "|---|---|---|---|---|---|\n"
    purposes = ["web extraction", "main analysis", "map extraction"]
    total_input = total_output = 0
    for i, call in enumerate(captured_calls):
        purpose = purposes[i] if i < len(purposes) else f"call_{i+1}"
        inp = call["usage"].get("prompt_tokens", 0)
        out = call["usage"].get("completion_tokens", 0)
        tot = call["usage"].get("total_tokens", inp + out)
        total_input += inp
        total_output += out
        meta_md += f"| {call['n']} | {purpose} | {inp} | {out} | {tot} | {call['duration_s']} |\n"
    meta_md += f"| **TOTAL** | — | **{total_input}** | **{total_output}** | **{total_input + total_output}** | — |\n"
    meta_md += "\n" + _h2("Wall-clock breakdown")
    llm_total = sum(c["duration_s"] for c in captured_calls)
    meta_md += _kv_table({
        "total_wall_clock_s": round(wall_clock_s, 2),
        "sum_of_llm_call_s": round(llm_total, 2),
        "non_llm_s (retrieval + render + parsing)":
            round(wall_clock_s - llm_total, 2),
    })
    p = out_dir / "10_pipeline_metadata.md"
    p.write_text(meta_md, encoding="utf-8")
    written.append(p)

    # --- map.png ----------------------------------------------------------
    if result.map is not None:
        map_path = out_dir / "map.png"
        try:
            result.map.savefig(
                map_path, dpi=150, bbox_inches="tight",
            )
            written.append(map_path)
        except Exception as exc:
            print(f"WARNING: failed to save map figure: {exc}")

    # --- README.md --------------------------------------------------------
    readme = "# Pipeline trace — Mexican avocado sustainability\n\n"
    readme += f"**Query:** {query}\n\n"
    readme += f"**Model:** `{model}`\n\n"
    readme += f"**Wall-clock:** {wall_clock_s:.1f}s | "
    readme += f"**LLM calls:** {len(captured_calls)} | "
    readme += f"**Total tokens:** {total_input + total_output}\n\n"
    readme += _h2("Artifacts (in pipeline order)")
    readme += "| # | File | Description |\n|---|---|---|\n"
    descs = [
        ("00", "00_run_config.md", "Config, env, git SHA, all assistant params"),
        ("01", "01_web_results_raw.md", "Raw web search results (pre-extraction)"),
        ("02", "02_llm_call_1_web_extraction.md", "LLM call #1: web structured extraction (request + response)"),
        ("03", "03_web_structured_signals.md", "Parsed output of call #1 (structured map signals)"),
        ("04", "04_rag_chunks.md", "8 retrieved RAG chunks with full text + scores"),
        ("05", "05_llm_call_2_main_analysis.md", "LLM call #2: main framework analysis (system + user + response)"),
        ("06", "06_parsed_analysis.md", "Parsed `ParsedAnalysis` fields from call #2's response"),
        ("07", "07_llm_call_3_map_extraction.md", "LLM call #3: structured map extraction (request + response)"),
        ("08", "08_map_data.md", "Parsed `map_data` dict (structured input to renderer)"),
        ("09", "09_formatted_output.md", "Final human-readable formatted text"),
        ("10", "10_pipeline_metadata.md", "Map type, notices, warnings, token usage table, wall-clock breakdown"),
        ("—", "map.png", "Rendered matplotlib figure (if `auto_map` produced one)"),
    ]
    for num, fname, desc in descs:
        readme += f"| {num} | [`{fname}`](./{fname}) | {desc} |\n"
    p = out_dir / "README.md"
    p.write_text(readme, encoding="utf-8")
    written.append(p)

    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    load_api_key()
    LoggingOpenAIAdapter = build_logging_adapter()

    from openai import OpenAI
    from metacouplingllm import MetacouplingAssistant

    print("=" * 60)
    print("METACOUPLING PIPELINE TRACE")
    print("=" * 60)
    print(f"  Query:  {QUERY}")
    print(f"  Model:  {MODEL}")
    print(f"  Out:    {OUT_DIR}")
    print()

    client = OpenAI()
    llm = LoggingOpenAIAdapter(client, model=MODEL)

    assistant = MetacouplingAssistant(
        llm_client=llm,
        rag_corpus="bundled",
        rag_max_chunks_per_paper=5,
        web_search=True,
        auto_map=True,
        verbose=True,
        coupling_analysis=True,
    )

    t0 = time.perf_counter()
    try:
        result = assistant.analyze(QUERY)
    except Exception as exc:
        print(f"\nFATAL: analyze() raised: {exc!r}")
        # Dump whatever we captured up to this point
        partial_dir = OUT_DIR / "PARTIAL_RUN"
        partial_dir.mkdir(parents=True, exist_ok=True)
        (partial_dir / "exception.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\nCaptured calls:\n"
            + json.dumps(
                [{"n": c["n"], "duration_s": c["duration_s"],
                  "response_chars": len(c["response_content"])}
                 for c in llm.captured_calls],
                indent=2,
            ),
            encoding="utf-8",
        )
        for i, call in enumerate(llm.captured_calls, start=1):
            (partial_dir / f"call_{i}_request.json").write_text(
                json.dumps({
                    "messages": call["messages"],
                    "temperature": call["temperature"],
                    "max_tokens": call["max_tokens"],
                }, indent=2),
                encoding="utf-8",
            )
            (partial_dir / f"call_{i}_response.txt").write_text(
                call["response_content"], encoding="utf-8",
            )
        print(f"Partial artifacts dumped to {partial_dir}")
        return 1
    wall_clock = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"  Analysis complete in {wall_clock:.1f}s")
    print(f"  LLM calls: {len(llm.captured_calls)}")
    print(f"  Map present: {result.map is not None}")
    print("=" * 60)
    print()

    written = write_artifacts(
        OUT_DIR,
        query=QUERY,
        model=MODEL,
        assistant=assistant,
        result=result,
        captured_calls=llm.captured_calls,
        wall_clock_s=wall_clock,
    )

    total_tokens = sum(
        c["usage"].get("total_tokens", 0) for c in llm.captured_calls
    )
    print(
        f"Wrote {len(written)} artifacts to {OUT_DIR} "
        f"({len(llm.captured_calls)} LLM calls, "
        f"{total_tokens} tokens, {wall_clock:.1f}s wall-clock)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
