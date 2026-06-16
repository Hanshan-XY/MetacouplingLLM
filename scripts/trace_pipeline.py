#!/usr/bin/env python3
"""Trace MetacouplingAssistant.analyze() end-to-end and dump every
intermediate artifact to disk for study.

As of the built-in run-tracing feature, the artifact-dumping logic lives in the
package (``metacouplingllm.tracing``); this script is now a thin driver that
loads an API key, constructs a ``MetacouplingAssistant`` with ``trace=True`` and
a fixed ``trace_dir``, and runs one query.  The artifacts (``00_run_config.md``
… ``10_pipeline_metadata.md`` + ``README.md`` + ``map.png``) are written under
``OUT_DIR/turn1/``.

Usage::

    python scripts/trace_pipeline.py

Edit ``QUERY``, ``OUT_DIR``, ``MODEL`` at the top to retarget.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

QUERY = "Impact of avocado production and trade in Mexico on sustainability"
OUT_DIR = Path("runs/avocado_2026-06-16_gpt-5.5_builtin-trace")
MODEL = "gpt-5.5"
API_KEY_PATH = Path(
    r"D:\Onedrive\OneDrive - Michigan State University\Desktop\api.env"
)


def load_api_key() -> None:
    """Read API key(s) from the configured .env file into the environment.

    Accepts either OPENAI_API_KEY or ANTHROPIC_API_KEY (whichever the
    configured MODEL targets).
    """
    if not API_KEY_PATH.exists():
        sys.exit(f"ERROR: API key file not found at {API_KEY_PATH}")
    for raw in API_KEY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ[key.strip()] = value.strip()
    if not (os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit(
            "ERROR: neither OPENAI_API_KEY nor ANTHROPIC_API_KEY found "
            f"in {API_KEY_PATH}"
        )


def main() -> int:
    load_api_key()

    from metacouplingllm import MetacouplingAssistant

    print("=" * 60)
    print("METACOUPLING PIPELINE TRACE")
    print("=" * 60)
    print(f"  Query:  {QUERY}")
    print(f"  Model:  {MODEL}")
    print(f"  Out:    {OUT_DIR}")
    print()

    # Plain provider adapters — the built-in trace wraps the client itself.
    if MODEL.startswith("claude"):
        import anthropic
        from metacouplingllm import AnthropicAdapter
        llm = AnthropicAdapter(anthropic.Anthropic(), model=MODEL)
    else:
        from openai import OpenAI
        from metacouplingllm import OpenAIAdapter
        llm = OpenAIAdapter(OpenAI(), model=MODEL)

    assistant = MetacouplingAssistant(
        llm_client=llm,
        rag_corpus="bundled",
        rag_max_chunks_per_paper=5,
        web_search=True,
        web_search_max_results=25,
        auto_map=True,
        verbose=True,
        coupling_analysis=True,
        trace=True,
        trace_dir=OUT_DIR,
    )

    t0 = time.perf_counter()
    result = assistant.analyze(QUERY)
    wall_clock = time.perf_counter() - t0

    trace = result.trace
    print()
    print("=" * 60)
    print(f"  Analysis complete in {wall_clock:.1f}s")
    if trace is not None:
        print(f"  LLM calls captured: {len(trace.calls)}")
        print(f"  Total tokens: {trace.total_tokens}")
        print(f"  Artifacts: {trace.out_dir}")
    else:
        print("  (no trace attached — tracing was disabled)")
    print(f"  Map present: {result.map is not None}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
