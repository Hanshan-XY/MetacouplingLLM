#!/usr/bin/env python
"""Generate the ADM1 English-exonym alias dataset.

Builds ``src/metacouplingllm/data/adm1_aliases.csv`` (the lean shipped table)
and ``docs/adm1_aliases_audit.csv`` (full provenance) by querying an LLM to
generate English exonyms / alternative spellings for all 3,373 ADM1 regions,
then applying deterministic validation rules.

This script is intended as a one-off offline build tool, NOT a library module.
The generated CSVs are committed to the repository; this script documents the
generation methodology and allows the dataset to be regenerated.

Generation approach
-------------------
Regions are grouped by country and batched into chunks of at most 20.  Each
chunk is sent to Claude as a single prompt requesting up to 3 English exonyms
per region.  Batching is concurrency-bounded to avoid API rate limits.

Alias scope (v1)
----------------
- English exonyms and alternative spellings only (e.g. "Bavaria" for "Bayern",
  "Tuscany" for "Toscana", "Mexico City" for "Ciudad de México").
- English romanizations when the canonical WB name uses non-Latin script.
- No abbreviations (NSW, CDMX), no historical/former names (Madras), no
  native non-Latin scripts.

Validation rules (deterministic, runs after generation)
--------------------------------------------------------
See ``scripts/validate_adm1_aliases.py`` for the full validation pipeline.
This script invokes it automatically after generation.

Usage
-----
    pip install anthropic  # if not already installed
    python scripts/build_adm1_aliases.py [--api-key KEY] [--model claude-sonnet-4-6] \\
        [--concurrency 8] [--out-dir build_data] [--chunk-size 20]

Run from the repository root. Set ANTHROPIC_API_KEY in the environment or
pass --api-key.  The final CSVs are written to their shipped paths.

Note: the first run takes ~10-20 minutes and costs approximately $3-8 USD
(depending on model and region count).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metacouplingllm.knowledge.adm1_pericoupling import _load_adm1_data  # noqa: E402

CHUNK_SIZE_DEFAULT = 20
CONCURRENCY_DEFAULT = 8
MODEL_DEFAULT = "claude-sonnet-4-6"


def _build_prompt(iso: str, country_name: str, chunk_index: int, regions: list[dict]) -> str:
    region_list = "\n".join(f"  {r['code']}: {r['name']}" for r in regions)
    return f"""You are generating English-exonym aliases for World Bank ADM1 regions.

Country: {country_name} ({iso})
Chunk: {chunk_index}

Regions:
{region_list}

For EACH region, generate up to 3 English exonyms or alternative spellings that:
- Differ materially from the canonical WB name shown
- Are commonly used in English academic or journalistic sources
- Are NOT abbreviations (NSW, BC, CDMX), historical names (Madras, Peking),
  or native non-Latin scripts

Skip a region if:
- The canonical name IS the standard English name
- You lack confidence (only "high" or "medium" allowed)
- Regions named "Name unknown"
- Any alias would be < 4 characters

For each alias provide:
  alias_display: proper capitalization (e.g. "Bavaria")
  alias_key: lowercase ASCII form of alias_display (e.g. "bavaria")
  basis: one-sentence explanation
  source: "pretraining" | "uncertain"
  confidence: "high" | "medium"

Return ONLY JSON:
{{
  "aliases": [
    {{
      "code": "DEU002",
      "iso_a3": "DEU",
      "alias_display": "Bavaria",
      "alias_key": "bavaria",
      "basis": "English exonym for the German state Bayern",
      "source": "pretraining",
      "confidence": "high"
    }}
  ]
}}
Return {{"aliases": []}} if no region in this batch warrants an alias."""


def _call_api(client, model: str, prompt: str, max_retries: int = 3) -> list[dict]:
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                return []
            data = json.loads(text[start:end])
            return data.get("aliases", [])
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  WARNING: API call failed after {max_retries} attempts: {exc}")
                return []
    return []


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_DEFAULT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--out-dir", default="build_data")
    args = parser.parse_args(argv)

    if not args.api_key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY or pass --api-key")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=args.api_key)
    except ImportError:
        sys.exit("ERROR: pip install anthropic")

    # Load regions
    _, _, _, metadata = _load_adm1_data()
    by_country: dict[str, list[dict]] = defaultdict(list)
    for code, m in sorted(metadata.items()):
        by_country[m["iso_a3"]].append({
            "code": code,
            "name": m["name"],
            "country_name": m["country_name"],
            "iso_a3": m["iso_a3"],
        })

    # Build chunks
    chunks: list[dict] = []
    for iso, regions in sorted(by_country.items()):
        country_name = regions[0]["country_name"]
        for i in range(0, len(regions), args.chunk_size):
            chunks.append({
                "iso": iso,
                "country_name": country_name,
                "chunk_index": i // args.chunk_size,
                "regions": regions[i:i + args.chunk_size],
            })

    print(f"Processing {len(chunks)} chunks ({len(metadata)} regions) "
          f"with concurrency={args.concurrency}, model={args.model}")

    raw_aliases: list[dict] = []

    def _process_chunk(chunk: dict) -> list[dict]:
        prompt = _build_prompt(
            chunk["iso"], chunk["country_name"], chunk["chunk_index"], chunk["regions"]
        )
        aliases = _call_api(client, args.model, prompt)
        return aliases

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_process_chunk, chunk): chunk for chunk in chunks}
        done = 0
        for future in as_completed(futures):
            aliases = future.result()
            raw_aliases.extend(aliases)
            done += 1
            if done % 20 == 0 or done == len(chunks):
                print(f"  {done}/{len(chunks)} chunks done, {len(raw_aliases)} raw candidates")

    # Save raw output
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_aliases.json"
    raw_path.write_text(json.dumps(raw_aliases, ensure_ascii=False, indent=None), encoding="utf-8")
    print(f"\nRaw aliases saved to {raw_path}")

    # Run validation
    print("\nRunning validation ...")
    validate_script = Path(__file__).parent / "validate_adm1_aliases.py"
    import subprocess
    result = subprocess.run(
        [sys.executable, str(validate_script), "--raw", str(raw_path)],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
