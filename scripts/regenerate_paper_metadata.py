"""Regenerate paper_citation_counts.csv from the rebuilt corpus.

One-off maintainer script.  Reflects the rebuilt Papers/ corpus
(420 papers: 296 OA full-text + 124 NOA structured summaries).

Strategy:
- Papers in both old CSV and current Papers/: preserve all old
  metadata (cited_by, doi, journal, keywords, etc.) so curated
  data is not lost.
- NOA papers new to corpus: parse metadata from the in-file
  structured-summary header (authoritative).
- OA papers new to corpus: look up metadata from BibTeX by title
  fuzzy match.
- Papers no longer in current corpus: dropped.

After this PR the CSV reflects the post-rebuild state.  Future
`cited_by` refreshes for new papers would still need a separate
Unpaywall / OpenAlex API call (see ``check_oa_status.py``).
"""

import csv
import re
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "Papers"
OLD_CSV = ROOT / "paper_citation_counts.csv"
NEW_CSV = ROOT / "paper_citation_counts.csv"  # in-place
BIB_PATH = ROOT / "src" / "metacouplingllm" / "data" / "telecoupling_literature.bib"

# Column order from old CSV
COLUMNS = [
    "filename", "authors", "year", "title", "category", "oa_status",
    "matched_bib_key", "journal", "doi", "keywords", "cited_by",
]


def load_old_csv():
    by_stem = {}
    with open(OLD_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = Path(row["filename"]).stem
            by_stem[stem] = row
    return by_stem


def parse_bib():
    """Return list of dicts; each entry has lowercase field names + '_key'."""
    text = BIB_PATH.read_text(encoding="utf-8")
    entry_pat = re.compile(r"@\w+\s*\{\s*([^,]+),(.*?)^\}", re.DOTALL | re.MULTILINE)
    field_pat = re.compile(r"(\w+)\s*=\s*\{(.*?)\}\s*,?", re.DOTALL)
    entries = []
    for m in entry_pat.finditer(text):
        key = m.group(1).strip()
        body = m.group(2)
        fields = {fm.group(1).lower(): fm.group(2).strip()
                  for fm in field_pat.finditer(body)}
        fields["_key"] = key
        entries.append(fields)
    return entries


def bib_match_for_stem(stem, bib_entries, min_overlap=0.6):
    """Find BibTeX entry whose title best matches the filename stem."""
    stem_norm = re.sub(r"[^a-z0-9 ]+", " ", stem.lower())
    stem_words = {w for w in stem_norm.split() if len(w) > 3}
    best, best_overlap = None, 0.0
    for entry in bib_entries:
        title = entry.get("title", "")
        title_norm = re.sub(r"[^a-z0-9 ]+", " ", title.lower())
        title_words = {w for w in title_norm.split() if len(w) > 3}
        if not title_words:
            continue
        overlap = len(title_words & stem_words) / len(title_words)
        if overlap > best_overlap and overlap >= min_overlap:
            best, best_overlap = entry, overlap
    return best


NOA_HEADER_PAT = re.compile(r"## Metadata\s*\n((?:- .+\n)+)", re.MULTILINE)


def parse_noa_header(text):
    """Return dict of YAML-ish key/value pairs from the NOA Metadata block,
    or None if the file is not an NOA structured-summary file."""
    if ("OA status: closed" not in text
            and "RAG record type: paraphrased" not in text):
        return None
    m = NOA_HEADER_PAT.search(text)
    if not m:
        return None
    out = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            k, _, v = line[2:].partition(":")
            out[k.strip()] = v.strip()
    return out


def build_new_rows(old_by_stem, bib_entries):
    new_rows = []
    n_preserved = 0
    n_noa_new = 0
    n_oa_new_matched = 0
    n_oa_new_unmatched = 0
    fn_pat = re.compile(r"^(.+?)\s*-\s*(\d{4})\s*-\s*(.+)$")

    for path in sorted(PAPERS_DIR.glob("*.md")):
        stem = path.stem
        filename = path.name

        # ALWAYS read the file first — the in-corpus header is
        # the authoritative source of OA status.  Note: the corpus
        # uses structured-summary format for 228 papers (copyright
        # safety), but only 124 are truly `closed`.  The remaining
        # 104 are gold / green / hybrid / bronze OA that just use
        # the same paraphrased format.  `non-OA` here means
        # specifically `OA status: closed`.
        text = path.read_text(encoding="utf-8", errors="replace")
        noa = parse_noa_header(text)
        oa_status_value = "OA"
        if noa and noa.get("OA status", "").strip() == "closed":
            oa_status_value = "non-OA"

        if stem in old_by_stem:
            row = OrderedDict(
                (c, old_by_stem[stem].get(c, "")) for c in COLUMNS
            )
            row["filename"] = filename
            # Override oa_status with the current corpus state.
            row["oa_status"] = oa_status_value
            # If summary header present, refresh doi/year/category
            # from header (more recent than old CSV).
            if noa:
                if "DOI" in noa and noa["DOI"]:
                    row["doi"] = noa["DOI"]
                if "Year" in noa and noa["Year"] not in ("", "unknown"):
                    row["year"] = noa["Year"]
                if "Category" in noa:
                    row["category"] = noa["Category"].split("\\")[-1].strip()
            new_rows.append(row)
            n_preserved += 1
            continue
        row = OrderedDict((c, "") for c in COLUMNS)
        row["filename"] = filename

        fn_match = fn_pat.match(stem)
        if fn_match:
            row["authors"] = fn_match.group(1).strip()
            row["year"] = fn_match.group(2)
            row["title"] = fn_match.group(3).strip()

        if noa:
            row["oa_status"] = oa_status_value  # "non-OA" only if header says closed
            if "DOI" in noa:
                row["doi"] = noa["DOI"]
            if "Category" in noa:
                cat = noa["Category"]
                # "Journal Articles\Research\Telecoupling" -> "Telecoupling"
                row["category"] = cat.split("\\")[-1].strip()
            if "Year" in noa and noa["Year"] not in ("", "unknown"):
                row["year"] = noa["Year"]
            n_noa_new += 1
        else:
            row["oa_status"] = "OA"
            entry = bib_match_for_stem(stem, bib_entries)
            if entry:
                row["matched_bib_key"] = entry.get("_key", "")
                row["journal"] = entry.get("journal", "")
                row["doi"] = entry.get("doi", "")
                row["keywords"] = entry.get("keywords", "")
                row["category"] = "Telecoupling"
                n_oa_new_matched += 1
            else:
                row["category"] = "Telecoupling"
                n_oa_new_unmatched += 1

        new_rows.append(row)

    print(f"Total new CSV rows: {len(new_rows)}")
    print(f"  Preserved from old CSV: {n_preserved}")
    print(f"  New NOA (from in-file header): {n_noa_new}")
    print(f"  New OA, BibTeX-matched: {n_oa_new_matched}")
    print(f"  New OA, no BibTeX match: {n_oa_new_unmatched}")
    return new_rows


def main():
    old_by_stem = load_old_csv()
    print(f"Old CSV: {len(old_by_stem)} rows")

    bib_entries = parse_bib()
    print(f"BibTeX: {len(bib_entries)} entries")
    print()

    new_rows = build_new_rows(old_by_stem, bib_entries)
    print()
    print(f"Final oa_status: {dict(Counter(r['oa_status'] for r in new_rows))}")
    print(f"Final category: {dict(Counter(r['category'] for r in new_rows))}")

    with open(NEW_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in new_rows:
            writer.writerow(row)

    print()
    print(f"Wrote {NEW_CSV}: {NEW_CSV.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
