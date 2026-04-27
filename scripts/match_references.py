"""
Match a curated CSV of papers against the bundled telecoupling_literature.bib.

For each CSV row, find the best matching .bib entry by (first-author surname,
year, title prefix) and report:
  1. CSV entries with no .bib match (need new BibTeX entries)
  2. Matched .bib entries that lack a "Cited by: N" annote (missing citation count)
  3. .bib entries NOT referenced by any CSV row (would be removed in an update)

Read-only. Writes nothing. Prints a structured report to stdout.

Usage:
    python scripts/match_references.py "<path/to/csv>"
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

# Force UTF-8 stdout so unicode in titles doesn't crash on Windows GBK consoles
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


# --------- BibTeX parsing (matches metacouplingllm.knowledge.literature) -------

_ENTRY_RE = re.compile(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", re.DOTALL)
_FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.*?)\}(?:\s*,)?", re.DOTALL)
_CITED_BY_RE = re.compile(r"Cited by:\s*(\d+)", re.IGNORECASE)


def parse_bib(path: Path) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    out: list[dict[str, str]] = []
    for m in _ENTRY_RE.finditer(text):
        body = m.group(3)
        fields: dict[str, str] = {
            "_key": m.group(2).strip(),
            "_raw": m.group(0),       # exact original text of the entry
            "_start": str(m.start()),  # byte offset for ordering
        }
        for fm in _FIELD_RE.finditer(body):
            name = fm.group(1).lower().strip()
            val = re.sub(r"\s+", " ", fm.group(2).strip())
            fields[name] = val
        if fields.get("year", "").isdigit():
            fields["year"] = fields["year"]
        out.append(fields)
    return out


def write_filtered_bib(
    bib_entries: list[dict[str, str]],
    keep_keys: set[str],
    output_path: Path,
) -> tuple[int, int]:
    """Write a new .bib containing only entries whose _key is in keep_keys.

    Preserves the exact original text of each kept entry and their original
    order in the source .bib. Returns (kept_count, removed_count).
    """
    kept: list[str] = []
    removed = 0
    for e in bib_entries:
        if e["_key"] in keep_keys:
            kept.append(e["_raw"])
        else:
            removed += 1
    output_path.write_text("\n\n".join(kept) + "\n", encoding="utf-8")
    return len(kept), removed


# --------- Normalisation helpers --------------------------------------------

_NONALNUM = re.compile(r"[^a-z0-9 ]+")
_MULTIWS = re.compile(r"\s+")
_HTML_TAG = re.compile(r"<[^>]+>")


def strip_diacritics(s: str) -> str:
    """Söndergaard -> Sondergaard, García -> Garcia, etc."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def norm_text(s: str) -> str:
    s = strip_diacritics(s or "")
    s = s.lower()
    s = s.replace("&amp;", "&").replace("\\&", "&")
    s = _HTML_TAG.sub("", s)               # strip <scp>X</scp>, <i>X</i>, etc.
    s = _NONALNUM.sub(" ", s)
    s = _MULTIWS.sub(" ", s).strip()
    return s


def all_surnames(authors: str) -> list[str]:
    """Return EVERY author surname in the entry (BibTeX or CSV form)."""
    a = (authors or "").strip()
    if not a:
        return []
    a = re.sub(r"\s+et\s+al\.?\s*$", "", a, flags=re.IGNORECASE).strip()
    parts = re.split(r"\s+and\s+", a, flags=re.IGNORECASE)
    surnames: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "," in p:
            surnames.append(norm_text(p.split(",")[0]))
        else:
            # CSV form like 'Bicudo da Silva' or 'Vila' — keep whole phrase
            surnames.append(norm_text(p))
    return [s for s in surnames if s]


def first_surname(authors: str) -> str:
    names = all_surnames(authors)
    return names[0] if names else ""


def surname_token_sets(authors: str) -> list[set[str]]:
    """Token-set per author surname (for fuzzy matching)."""
    return [set(s.split()) for s in all_surnames(authors)]


def title_prefix(title: str, n_words: int = 6) -> str:
    """First N words of a normalised title (handles CSV truncation)."""
    return " ".join(norm_text(title).split()[:n_words])


# --------- Matching ----------------------------------------------------------

def _title_score(csv_title_norm: str, bib_title_norm: str) -> int:
    """Score 0-N for how many leading words of CSV title match .bib title.

    Also returns a bonus score if the CSV's first 4 normalized words
    appear as a substring anywhere in the .bib title (handles cases
    where the CSV title omits a leading word).
    """
    if not csv_title_norm or not bib_title_norm:
        return 0
    c_words = csv_title_norm.split()
    b_words = bib_title_norm.split()
    common = 0
    for cw, bw in zip(c_words, b_words):
        if cw == bw:
            common += 1
        else:
            break
    if common >= min(4, len(c_words)):
        return common
    pref4 = " ".join(c_words[:4])
    if pref4 and pref4 in bib_title_norm:
        return 3
    # Tail-fallback: CSV may have prefix like "From X with love" while
    # .bib has "From Kenya with love" — score the longest matching
    # subsequence of leading 6 words
    pref_any = " ".join(c_words[:6])
    if pref_any and any(pref_any[: i + 1] in bib_title_norm for i in range(15, len(pref_any))):
        return 2
    return 0


def find_match(csv_authors: str, csv_year: str, csv_title: str,
               bib_entries: list[dict[str, str]]) -> dict[str, str] | None:
    """Return best-matching .bib entry, or None.

    Match rule (year-strict + author-fuzzy + title-prefix):
      - year matches exactly (or within ±1 to handle CSV/.bib year drift)
      - ANY surname in CSV overlaps with ANY surname in .bib (token-set
        intersection, after diacritics stripping)
      - normalised CSV title prefix matches normalised .bib title

    Falls back to (year + strong title prefix) when author match fails.
    """
    csv_year = csv_year.strip()
    csv_surname_sets = surname_token_sets(csv_authors)
    csv_norm_title = norm_text(csv_title)

    candidates: list[tuple[int, dict[str, str]]] = []
    for e in bib_entries:
        bib_year = e.get("year", "").strip()
        if not bib_year or not csv_year:
            year_ok = False
        else:
            try:
                year_ok = abs(int(csv_year) - int(bib_year)) <= 1
            except ValueError:
                year_ok = csv_year == bib_year
        if not year_ok:
            continue

        bib_surname_sets = surname_token_sets(e.get("author", ""))
        # Author overlap: does ANY csv surname intersect ANY bib surname?
        author_match = any(
            cs & bs for cs in csv_surname_sets for bs in bib_surname_sets
        )

        bib_norm_title = norm_text(e.get("title", ""))
        title_score = _title_score(csv_norm_title, bib_norm_title)

        if author_match and title_score >= 2:
            # Strong: author + decent title match
            candidates.append((title_score + 10, e))
        elif title_score >= 5:
            # Weak author but strong title prefix → still likely a match
            candidates.append((title_score, e))
        elif author_match and title_score >= 1:
            # Weak title but author hit → probable match
            candidates.append((title_score + 5, e))

    if not candidates:
        return None
    candidates.sort(key=lambda kv: -kv[0])
    return candidates[0][1]


# --------- Reporting ---------------------------------------------------------

def main(csv_path: Path, bib_path: Path, write_to: Path | None = None) -> int:
    bib_entries = parse_bib(bib_path)
    print(f"Loaded {len(bib_entries)} entries from {bib_path.name}")

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)
    print(f"Loaded {len(csv_rows)} entries from {csv_path.name}\n")

    matched: list[tuple[dict[str, str], dict[str, str]]] = []
    unmatched: list[dict[str, str]] = []
    matched_bib_keys: set[str] = set()

    for row in csv_rows:
        csv_authors = row.get("Authors", "").strip()
        csv_year = row.get("Year", "").strip()
        csv_title = row.get("Title", "").strip()
        match = find_match(csv_authors, csv_year, csv_title, bib_entries)
        if match is None:
            unmatched.append(row)
        else:
            matched.append((row, match))
            matched_bib_keys.add(match["_key"])

    # Report 1: CSV entries with no .bib match
    print("=" * 78)
    print(f"REPORT 1 - CSV entries NOT matched in .bib  ({len(unmatched)} of {len(csv_rows)})")
    print("=" * 78)
    if not unmatched:
        print("(none - every CSV entry matched a .bib entry)")
    for r in unmatched:
        print(f"  - [{r.get('Category','?')}] {r.get('Authors','?')} ({r.get('Year','?')}) - "
              f"{(r.get('Title','') or '')[:90]}")

    # Report 2: matched entries with no Cited-by annotation
    print()
    print("=" * 78)
    print(f"REPORT 2 - Matched .bib entries with NO citation count  "
          f"(of {len(matched)} matched)")
    print("=" * 78)
    no_cite = []
    for csv_row, bib in matched:
        annote = bib.get("annote", "")
        if not _CITED_BY_RE.search(annote):
            no_cite.append((csv_row, bib))
    if not no_cite:
        print("(none - every matched .bib entry has a Cited-by annotation)")
    for csv_row, bib in no_cite:
        print(f"  - [{csv_row.get('Category','?')}] {csv_row.get('Authors','?')} "
              f"({csv_row.get('Year','?')})")
        print(f"    .bib key: {bib['_key']}")
        print(f"    title:    {(bib.get('title','') or '')[:90]}")

    # Report 3: .bib entries not referenced by CSV (would be removed)
    unreferenced = [e for e in bib_entries if e["_key"] not in matched_bib_keys]
    print()
    print("=" * 78)
    print(f"REPORT 3 - .bib entries NOT referenced by CSV  "
          f"({len(unreferenced)} of {len(bib_entries)} would be removed)")
    print("=" * 78)
    for e in unreferenced[:50]:
        print(f"  - {e['_key']}: {e.get('author','?')[:60]} ({e.get('year','?')}) - "
              f"{(e.get('title','') or '')[:80]}")
    if len(unreferenced) > 50:
        print(f"  ... and {len(unreferenced) - 50} more")

    # Final summary
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  CSV rows:                        {len(csv_rows)}")
    print(f"  Matched to .bib:                 {len(matched)}")
    print(f"  CSV unmatched (need new BibTeX): {len(unmatched)}")
    print(f"  Matched but missing cited_by:    {len(no_cite)}")
    print(f"  .bib entries to remove:          {len(unreferenced)}")

    if write_to is not None:
        kept, removed = write_filtered_bib(
            bib_entries, matched_bib_keys, write_to,
        )
        print()
        print("=" * 78)
        print(f"WROTE filtered .bib to {write_to}")
        print(f"  Kept:    {kept} entries")
        print(f"  Removed: {removed} entries")
        print("=" * 78)
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Match a CSV of papers against telecoupling_literature.bib",
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument(
        "--bib", type=Path, default=None,
        help="Path to .bib file (default: bundled telecoupling_literature.bib)",
    )
    parser.add_argument(
        "--write", type=Path, default=None,
        help="If given, write a new .bib to this path containing only matched entries",
    )
    args = parser.parse_args()

    bib_p = args.bib or (
        Path(__file__).resolve().parent.parent
        / "src" / "metacoupling" / "data" / "telecoupling_literature.bib"
    )
    sys.exit(main(args.csv_path, bib_p, write_to=args.write))
