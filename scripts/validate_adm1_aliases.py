#!/usr/bin/env python
"""Validate and write the ADM1 alias CSVs from raw workflow output.

Reads build_data/raw_aliases.json (output of the alias generation workflow),
applies all deterministic validation rules, and writes:

  src/metacouplingllm/data/adm1_aliases.csv   — lean shipped table
  docs/adm1_aliases_audit.csv                 — full provenance table

Validation rules
----------------
N0  Normalize key: lowercase, strip, collapse spaces, NFKD-fold.
G   Drop if the key resolves as a country name.
S1  Drop keys shorter than 4 characters.
W1  Drop generic single geo-words (direction/type words, ADM1 suffixes).
R1  Drop if resolve_adm1_code(key, country=iso) already returns the same code
    (redundant — the existing resolver already handles it).
O1  If the resolver returns a DIFFERENT non-None code → mark as override;
    exclude from the shipped CSV, list in the audit for manual review.
U1  Intra-country: if two aliases in the same country share a key → drop both.
U2  Cross-country: if a key maps to regions in more than one country → drop
    (global key ambiguity; callers would need a country hint, reducing value).
U3  Drop if key matches an existing canonical index entry for a different code
    (would shadow a real region name).
C1  Cap at 3 aliases per code (highest confidence first, then alphabetical).

Usage
-----
    python scripts/validate_adm1_aliases.py [--raw build_data/raw_aliases.json]

Run after the alias-generation workflow completes and
build_data/raw_aliases.json has been written.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metacouplingllm.knowledge.adm1_pericoupling import (  # noqa: E402
    _load_adm1_data,
    _get_adm1_name_index,
    resolve_adm1_code,
)
from metacouplingllm.knowledge.countries import resolve_country_code  # noqa: E402

# ---------------------------------------------------------------------------
# Generic geo-words to drop (W1)
# ---------------------------------------------------------------------------
_W1_WORDS: frozenset[str] = frozenset({
    "north", "south", "east", "west", "central", "upper", "lower",
    "new", "old", "great", "greater", "inner", "outer",
    "region", "province", "state", "city", "county", "district",
    "department", "territory", "island", "islands", "coast",
    "sheng", "oblast", "okrug", "kray", "pradesh", "laen",
    "novads", "maakond", "zizhiqu", "rep", "giang",
})


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _normalize_key(raw: str) -> str:
    """N0: lowercase, strip, collapse spaces, NFKD-fold."""
    return _fold(raw.strip().lower())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="build_data/raw_aliases.json")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        sys.exit(f"ERROR: {raw_path} not found — run the generation workflow first")

    print(f"Loading raw aliases from {raw_path} ...")
    with open(raw_path, encoding="utf-8") as fh:
        raw_aliases: list[dict] = json.load(fh)
    print(f"  {len(raw_aliases)} raw candidates")

    # Load ADM1 data for validation
    print("Loading ADM1 metadata ...")
    _, _, _, adm1_metadata = _load_adm1_data()
    name_index = _get_adm1_name_index()
    valid_codes = set(adm1_metadata.keys())

    # Prevent circular contamination: any previously-shipped adm1_aliases.csv
    # would be loaded by _ensure_loaded() and used in Strategy 0 during R1
    # checks, making the R1 check self-referential.  Clear it so R1 tests the
    # pure name/substring resolver.
    import metacouplingllm.knowledge.adm1_pericoupling as _adm1_mod
    _adm1_mod._adm1_aliases = {}

    # ---------------------------------------------------------------------------
    # Per-candidate normalization + per-rule filtering
    # ---------------------------------------------------------------------------
    records: list[dict] = []
    drop_counts: dict[str, int] = defaultdict(int)

    for raw in raw_aliases:
        code = raw.get("code", "").strip()
        iso = raw.get("iso_a3", "").strip()
        alias_display = raw.get("alias_display", "").strip()
        alias_key_raw = raw.get("alias_key", "").strip()
        basis = raw.get("basis", "").strip()
        source = raw.get("source", "pretraining").strip()
        confidence = raw.get("confidence", "medium").strip()

        if code not in valid_codes:
            drop_counts["invalid_code"] += 1
            continue

        # N0
        key = _normalize_key(alias_key_raw or alias_display)
        if not key:
            drop_counts["N0_empty"] += 1
            continue

        # S1
        if len(key) < 4:
            drop_counts["S1_short"] += 1
            continue

        # G
        if resolve_country_code(key) is not None:
            drop_counts["G_country_name"] += 1
            continue

        # W1
        if key in _W1_WORDS:
            drop_counts["W1_generic"] += 1
            continue

        # R1 / O1
        resolved = resolve_adm1_code(key, country=iso)
        if resolved == code:
            drop_counts["R1_redundant"] += 1
            continue
        override = False
        if resolved is not None and resolved != code:
            override = True

        records.append({
            "code": code,
            "iso_a3": iso,
            "alias_display": alias_display,
            "alias_key": key,
            "basis": basis,
            "source": source,
            "confidence": confidence,
            "override": override,
        })

    print(f"  After N0/G/S1/W1/R1/O1: {len(records)} candidates remain")
    for rule, count in sorted(drop_counts.items()):
        print(f"    dropped by {rule}: {count}")

    # ---------------------------------------------------------------------------
    # U1: intra-country key uniqueness
    # ---------------------------------------------------------------------------
    # key × country → list of codes
    intra: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in records:
        intra[(rec["alias_key"], rec["iso_a3"])].append(rec["code"])

    u1_collisions: set[tuple[str, str]] = {k for k, v in intra.items() if len(set(v)) > 1}
    before_u1 = len(records)
    records = [r for r in records if (r["alias_key"], r["iso_a3"]) not in u1_collisions]
    drop_counts["U1_intra"] = before_u1 - len(records)
    print(f"  After U1 (intra-country uniqueness): {len(records)}")

    # ---------------------------------------------------------------------------
    # U2: cross-country key uniqueness
    # ---------------------------------------------------------------------------
    cross: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        cross[rec["alias_key"]].add(rec["iso_a3"])

    u2_ambiguous: set[str] = {k for k, isos in cross.items() if len(isos) > 1}
    before_u2 = len(records)
    records = [r for r in records if r["alias_key"] not in u2_ambiguous]
    drop_counts["U2_cross"] = before_u2 - len(records)
    print(f"  After U2 (cross-country uniqueness): {len(records)}")

    # ---------------------------------------------------------------------------
    # U3: don't shadow an existing canonical name for a different code
    # ---------------------------------------------------------------------------
    before_u3 = len(records)
    kept_u3 = []
    for rec in records:
        existing = name_index.get(rec["alias_key"])
        if existing:
            existing_codes = {c for c, _ in existing}
            if existing_codes != {rec["code"]}:
                drop_counts["U3_shadows"] += 1
                continue
        kept_u3.append(rec)
    records = kept_u3
    drop_counts.setdefault("U3_shadows", 0)
    print(f"  After U3 (no shadow): {len(records)}")

    # ---------------------------------------------------------------------------
    # D0: deduplicate identical (code, alias_key) pairs produced by parallel agents
    # ---------------------------------------------------------------------------
    seen_pairs: set[tuple[str, str]] = set()
    deduped = []
    for rec in records:
        pair = (rec["code"], rec["alias_key"])
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            deduped.append(rec)
    drop_counts["D0_duplicate"] = len(records) - len(deduped)
    records = deduped
    print(f"  After D0 (dedup same code+key): {len(records)}")

    # ---------------------------------------------------------------------------
    # C1: cap at 3 per code (high > medium; pretraining > uncertain; alphabetical)
    # ---------------------------------------------------------------------------
    def _sort_key(r: dict) -> tuple:
        conf_order = {"high": 0, "medium": 1}.get(r["confidence"], 2)
        src_order = {"pretraining": 0, "uncertain": 1}.get(r["source"], 2)
        return (conf_order, src_order, r["alias_display"])

    by_code: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_code[rec["code"]].append(rec)

    final_shipped: list[dict] = []
    final_audit: list[dict] = []
    for code_key, recs in by_code.items():
        recs_sorted = sorted(recs, key=_sort_key)
        # Split overrides out (not shipped)
        non_override = [r for r in recs_sorted if not r["override"]]
        override_recs = [r for r in recs_sorted if r["override"]]
        shipped = non_override[:3]
        final_shipped.extend(shipped)
        # Audit includes everything (overrides flagged)
        final_audit.extend(recs_sorted)

    print(f"\nFinal shipped aliases: {len(final_shipped)}")
    override_count = sum(1 for r in final_audit if r["override"])
    print(f"Override candidates (excluded from shipped, in audit): {override_count}")
    print(f"Codes with at least one alias: {len(by_code)}")

    # ---------------------------------------------------------------------------
    # Write shipped CSV
    # ---------------------------------------------------------------------------
    shipped_path = Path("src/metacouplingllm/data/adm1_aliases.csv")
    shipped_path.parent.mkdir(parents=True, exist_ok=True)
    shipped_fields = ["alias_key", "alias_display", "code", "iso_a3"]
    with open(shipped_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=shipped_fields)
        w.writeheader()
        for rec in sorted(final_shipped, key=lambda r: (r["code"], r["alias_key"])):
            w.writerow({f: rec[f] for f in shipped_fields})
    print(f"Written: {shipped_path}")

    # ---------------------------------------------------------------------------
    # Write audit CSV
    # ---------------------------------------------------------------------------
    audit_path = Path("docs/adm1_aliases_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_fields = ["alias_key", "alias_display", "code", "iso_a3", "basis", "source", "confidence", "override"]
    with open(audit_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        for rec in sorted(final_audit, key=lambda r: (r["code"], r["alias_key"])):
            w.writerow({f: rec.get(f, "") for f in audit_fields})
    print(f"Written: {audit_path}")

    # ---------------------------------------------------------------------------
    # Final assertions
    # ---------------------------------------------------------------------------
    print("\nRunning final assertions ...")
    shipped_keys = [r["alias_key"] for r in final_shipped]
    assert len(shipped_keys) == len(set(shipped_keys)), "FAIL: duplicate alias_key in shipped CSV"
    for rec in final_shipped:
        assert rec["code"] in valid_codes, f"FAIL: unknown code {rec['code']}"
        assert resolve_country_code(rec["alias_key"]) is None, f"FAIL: alias_key is a country name: {rec['alias_key']}"
    print("All assertions passed.")

    # Summary by drop rule
    print("\nDrop summary:")
    total_dropped = len(raw_aliases) - len(final_shipped)
    for rule, count in sorted(drop_counts.items()):
        print(f"  {rule}: {count}")
    print(f"  Total dropped: {total_dropped} / {len(raw_aliases)}")


if __name__ == "__main__":
    main()
