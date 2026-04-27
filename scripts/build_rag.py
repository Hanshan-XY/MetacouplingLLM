#!/usr/bin/env python3
"""Build RAG corpus from PDF collection.

Steps:
1. Parse metadata from Zotero-style PDF filenames
2. Scan PDFs for DOIs using PyMuPDF
3. Enrich metadata via CrossRef API (free, no key needed)
4. Build new BibTeX file
5. Convert OA PDFs to full-text markdown

Usage:
    python scripts/build_rag.py

Inputs:
    - rag_build/oa/       — 170 Open Access PDFs
    - rag_build/non_oa/   — 92 non-OA PDFs (only metadata extracted, no full text)
    - paper_citation_counts_filled.csv — user-provided citation counts

Outputs:
    - src/metacoupling/data/telecoupling_literature.bib  — rebuilt BibTeX
    - Papers/*.md  — full-text markdown (OA) or placeholder (non-OA)
"""
from __future__ import annotations

import csv
import os
import re
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
OA_DIR = BASE_DIR / "rag_build" / "oa"
NONOA_DIR = BASE_DIR / "rag_build" / "non_oa"
OUTPUT_DIR = BASE_DIR / "Papers"
BIB_PATH = BASE_DIR / "src" / "metacoupling" / "data" / "telecoupling_literature.bib"
CSV_PATH = Path("C:/Users/rzyx1/Downloads/paper_citation_counts_filled.csv")

# DOI regex
DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s\]>\"',;]+)")

# Filename regex (Zotero-style)
FILENAME_RE = re.compile(
    r"^(?P<authors>.+?)\s*-\s*(?P<year>\d{4})\s*-\s*(?P<title>.+)\.pdf$"
)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_citation_csv() -> dict[str, dict]:
    """Load user-provided citation counts CSV. Keyed by filename."""
    rows: dict[str, dict] = {}
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows[row["filename"]] = row
    return rows


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def parse_filename(pdf_name: str) -> dict[str, str]:
    """Extract author, year, title from Zotero filename."""
    m = FILENAME_RE.match(pdf_name)
    if m:
        return {
            "authors": m.group("authors"),
            "year": m.group("year"),
            "title": m.group("title"),
        }
    return {"authors": "", "year": "", "title": pdf_name.replace(".pdf", "")}


def extract_doi(pdf_path: Path) -> str | None:
    """Scan first 3 pages of a PDF for a DOI."""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for i in range(min(3, doc.page_count)):
            text += doc[i].get_text()
        doc.close()

        matches = DOI_RE.findall(text)
        if matches:
            # Clean trailing punctuation
            doi = matches[0].rstrip(".,;:)")
            return doi
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# CrossRef enrichment
# ---------------------------------------------------------------------------

def query_crossref(doi: str) -> dict | None:
    """Query CrossRef for paper metadata. Returns dict or None."""
    import requests

    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "metacoupling-rag-builder/1.0 "
        "(mailto:metacoupling@example.com)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json().get("message", {})
            result = {}

            # Title
            titles = data.get("title", [])
            if titles:
                result["title"] = titles[0]

            # Authors
            authors = data.get("author", [])
            author_strs = []
            for a in authors:
                parts = []
                if a.get("given"):
                    parts.append(a["given"])
                if a.get("family"):
                    parts.append(a["family"])
                if parts:
                    author_strs.append(", ".join(reversed(parts)))
            if author_strs:
                result["authors"] = " and ".join(author_strs)

            # Journal
            journals = data.get("container-title", [])
            if journals:
                result["journal"] = journals[0]

            # Volume, issue, pages
            if data.get("volume"):
                result["volume"] = data["volume"]
            if data.get("issue"):
                result["number"] = data["issue"]
            if data.get("page"):
                result["pages"] = data["page"]

            # Abstract
            if data.get("abstract"):
                # CrossRef abstracts have JATS XML tags
                abstract = re.sub(r"<[^>]+>", "", data["abstract"])
                result["abstract"] = abstract.strip()

            # Keywords / subjects
            subjects = data.get("subject", [])
            if subjects:
                result["keywords"] = [s.lower() for s in subjects]

            # Publisher
            if data.get("publisher"):
                result["publisher"] = data["publisher"]

            # Year
            issued = data.get("issued", {}).get("date-parts", [[]])
            if issued and issued[0] and issued[0][0]:
                result["year"] = str(issued[0][0])

            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# BibTeX generation
# ---------------------------------------------------------------------------

def make_bib_key(authors: str, year: str, title: str) -> str:
    """Generate a BibTeX key like 'liu_telecoupling_2013'."""
    # First author surname
    first = authors.split(" and ")[0] if " and " in authors else authors
    # Handle "LastName, FirstName" format
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1] if first else "unknown"
    surname = re.sub(r"[^a-z]", "", surname.lower())

    # First significant word from title
    stop = {"the", "a", "an", "of", "in", "on", "for", "and", "to", "from", "with", "by"}
    words = re.findall(r"[a-z]+", title.lower())
    keyword = next((w for w in words if w not in stop and len(w) > 2), "paper")

    return f"{surname}_{keyword}_{year}"


def escape_bibtex(s: str) -> str:
    """Escape special BibTeX characters."""
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def format_bib_entry(entry: dict) -> str:
    """Format a single BibTeX entry."""
    key = entry["key"]
    fields = []

    if entry.get("title"):
        fields.append(f"  title = {{{entry['title']}}}")
    if entry.get("authors"):
        fields.append(f"  author = {{{entry['authors']}}}")
    if entry.get("year"):
        fields.append(f"  year = {{{entry['year']}}}")
    if entry.get("journal"):
        fields.append(f"  journal = {{{entry['journal']}}}")
    if entry.get("volume"):
        fields.append(f"  volume = {{{entry['volume']}}}")
    if entry.get("number"):
        fields.append(f"  number = {{{entry['number']}}}")
    if entry.get("pages"):
        fields.append(f"  pages = {{{entry['pages']}}}")
    if entry.get("doi"):
        fields.append(f"  doi = {{{entry['doi']}}}")
    if entry.get("abstract"):
        abstract = entry["abstract"][:2000]  # Truncate very long abstracts
        fields.append(f"  abstract = {{{abstract}}}")
    # Ensure every paper has at least one coupling-related keyword
    # (all papers in this collection are curated telecoupling/metacoupling studies)
    keywords = list(entry.get("keywords", []))
    coupling_terms = {"telecoupling", "tele-coupling", "metacoupling",
                      "meta-coupling", "pericoupling", "intracoupling",
                      "coupled human and natural", "coupled systems"}
    has_coupling_kw = any(t in " ".join(keywords).lower() for t in coupling_terms)
    if not has_coupling_kw:
        # Detect from category or title
        cat = entry.get("category_label", "").lower()
        title_lower = entry.get("title", "").lower()
        if "metacoupling" in cat or "metacoupling" in title_lower or "meta-coupling" in title_lower:
            keywords.append("metacoupling")
        elif "telecoupling" in title_lower or "tele-coupling" in title_lower or "telecouple" in title_lower:
            keywords.append("telecoupling")
        else:
            keywords.append("telecoupling")  # default — all are from coupling collection
    if keywords:
        kw = ", ".join(keywords)
        fields.append(f"  keywords = {{{kw}}}")
    if entry.get("publisher"):
        fields.append(f"  publisher = {{{entry['publisher']}}}")

    # Citation count
    cited = entry.get("cited_by", 0)
    if cited:
        fields.append(f"  annote = {{Cited by: {cited}}}")

    body = ",\n".join(fields)
    return f"@article{{{key},\n{body}\n}}\n"


# ---------------------------------------------------------------------------
# PDF → Markdown conversion (OA papers)
# ---------------------------------------------------------------------------

# Section heading heuristics
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s\-&:,]{3,80}$")
_NUMBERED_HEADING_RE = re.compile(r"^\d+\.?\s+[A-Z]")

# Sections to exclude
_EXCLUDE_SECTIONS = {
    "references", "bibliography", "literature cited",
    "acknowledgments", "acknowledgements", "acknowledgment",
    "appendix", "supplementary material", "supporting information",
    "supplementary data", "author contributions", "conflict of interest",
    "conflicts of interest", "declaration of competing interest",
    "data availability", "credit authorship contribution statement",
    "funding", "declaration of interests",
}


def pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a PDF to structured markdown text."""
    doc = fitz.open(str(pdf_path))
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text())
    doc.close()

    lines: list[str] = []
    in_excluded = False

    for page_text in pages_text:
        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            # Detect section headings
            is_heading = False
            heading_text = stripped

            # All-caps headings (e.g., "INTRODUCTION", "METHODS")
            if _ALLCAPS_HEADING_RE.match(stripped) and len(stripped.split()) <= 8:
                is_heading = True
                heading_text = stripped.title()

            # Numbered headings (e.g., "1. Introduction", "2.1 Study Area")
            elif _NUMBERED_HEADING_RE.match(stripped) and len(stripped) < 100:
                is_heading = True
                heading_text = stripped

            if is_heading:
                # Check if this is an excluded section
                check = heading_text.lower().strip()
                # Remove leading numbers
                check = re.sub(r"^\d+\.?\d*\.?\s*", "", check)
                if check in _EXCLUDE_SECTIONS:
                    in_excluded = True
                    continue
                else:
                    in_excluded = False
                    lines.append(f"\n## {heading_text}\n")
            elif not in_excluded:
                lines.append(stripped)

    # Join and clean up
    text = "\n".join(lines)
    # Merge broken lines (lines that don't end with sentence-enders)
    text = re.sub(r"(?<=[a-z,])\n(?=[a-z])", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RAG CORPUS BUILDER")
    print("=" * 60)

    # Load CSV
    print("\n[1/5] Loading citation counts CSV...")
    csv_data = load_citation_csv()
    print(f"  Loaded {len(csv_data)} entries")

    # Collect all PDFs
    print("\n[2/5] Scanning PDFs and extracting DOIs...")
    entries: list[dict] = []
    all_pdfs: list[tuple[Path, str]] = []  # (path, oa_status)

    for pdf_file in sorted(OA_DIR.glob("*.pdf")):
        all_pdfs.append((pdf_file, "oa"))
    for pdf_file in sorted(NONOA_DIR.glob("*.pdf")):
        all_pdfs.append((pdf_file, "non_oa"))

    print(f"  Found {len(all_pdfs)} PDFs ({sum(1 for _, s in all_pdfs if s == 'oa')} OA, "
          f"{sum(1 for _, s in all_pdfs if s == 'non_oa')} non-OA)")

    for pdf_path, oa_status in all_pdfs:
        fname = pdf_path.name
        meta = parse_filename(fname)
        doi = extract_doi(pdf_path)

        # Get citation count from CSV
        csv_row = csv_data.get(fname, {})
        cited_str = csv_row.get("cited_by", "").strip()
        cited_by = int(cited_str) if cited_str and cited_str.isdigit() else 0

        # Get existing keywords from CSV (from old BibTeX)
        kw_str = csv_row.get("keywords", "")
        keywords = [k.strip() for k in kw_str.split(";") if k.strip()] if kw_str else []

        # Get journal from CSV
        journal = csv_row.get("journal", "")

        entries.append({
            "filename": fname,
            "oa_status": oa_status,
            "pdf_path": str(pdf_path),
            "authors": meta["authors"],
            "year": meta["year"],
            "title": meta["title"],
            "doi": doi,
            "cited_by": cited_by,
            "keywords": keywords,
            "journal": journal,
            "abstract": "",
            "volume": "",
            "number": "",
            "pages": "",
            "publisher": "",
            "category_label": csv_row.get("category", ""),
        })

    dois_found = sum(1 for e in entries if e["doi"])
    print(f"  DOIs found: {dois_found}/{len(entries)}")

    # CrossRef enrichment
    print(f"\n[3/5] Enriching metadata via CrossRef ({dois_found} DOIs)...")
    enriched = 0
    for i, entry in enumerate(entries):
        if not entry["doi"]:
            continue

        cr = query_crossref(entry["doi"])
        if cr:
            enriched += 1
            # Only override if CrossRef has better data
            if cr.get("title") and len(cr["title"]) > len(entry["title"]):
                entry["title"] = cr["title"]
            if cr.get("authors"):
                entry["authors"] = cr["authors"]
            if cr.get("journal"):
                entry["journal"] = cr["journal"]
            if cr.get("abstract") and not entry["abstract"]:
                entry["abstract"] = cr["abstract"]
            if cr.get("keywords") and not entry["keywords"]:
                entry["keywords"] = cr["keywords"]
            if cr.get("volume"):
                entry["volume"] = cr["volume"]
            if cr.get("number"):
                entry["number"] = cr["number"]
            if cr.get("pages"):
                entry["pages"] = cr["pages"]
            if cr.get("publisher"):
                entry["publisher"] = cr["publisher"]

        # Rate limit: CrossRef polite pool
        if (i + 1) % 20 == 0:
            print(f"    ... {i + 1}/{dois_found} queried, {enriched} enriched")
            time.sleep(1)

    print(f"  Enriched {enriched}/{dois_found} papers from CrossRef")

    # Generate BibTeX keys (ensure uniqueness)
    print("\n[4/5] Building BibTeX and converting OA papers to markdown...")
    seen_keys: set[str] = set()
    for entry in entries:
        key = make_bib_key(entry["authors"], entry["year"], entry["title"])
        # Ensure uniqueness
        base_key = key
        counter = 1
        while key in seen_keys:
            key = f"{base_key}_{chr(96 + counter)}"  # a, b, c, ...
            counter += 1
        seen_keys.add(key)
        entry["key"] = key

    # Write BibTeX
    # Backup old BibTeX first
    if BIB_PATH.exists():
        backup = BIB_PATH.with_suffix(".bib.bak")
        BIB_PATH.rename(backup)
        print(f"  Backed up old BibTeX to {backup.name}")

    with open(BIB_PATH, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(format_bib_entry(entry))
            f.write("\n")
    print(f"  Wrote {len(entries)} BibTeX entries to {BIB_PATH.name}")

    # Convert OA PDFs to markdown
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oa_converted = 0
    for entry in entries:
        if entry["oa_status"] != "oa":
            continue

        pdf_path = Path(entry["pdf_path"])
        md_name = entry["filename"].replace(".pdf", ".md")
        md_path = OUTPUT_DIR / md_name

        try:
            md_text = pdf_to_markdown(pdf_path)
            if len(md_text.split()) < 100:
                print(f"  WARNING: Very short extraction for {md_name} "
                      f"({len(md_text.split())} words)")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            oa_converted += 1
        except Exception as exc:
            print(f"  ERROR converting {entry['filename']}: {exc}")

    print(f"  Converted {oa_converted}/170 OA papers to markdown")

    # Summary
    print("\n[5/5] Summary")
    print(f"  BibTeX entries: {len(entries)}")
    print(f"  OA markdown files: {oa_converted}")
    print(f"  Non-OA PDFs (awaiting Claude summaries): "
          f"{sum(1 for e in entries if e['oa_status'] == 'non_oa')}")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  BibTeX file: {BIB_PATH}")
    print(f"\nNext step: Run Claude agent to summarize 92 non-OA PDFs")


if __name__ == "__main__":
    main()
