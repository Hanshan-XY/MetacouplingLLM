"""Extract keywords from full-text markdown papers and match to .bib keys.

One-off script — not part of the shipped package.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import tempfile

MD_ROOT = r"D:/Check/_markdown/Journal Articles/Research"
BIB_PATH = "src/metacoupling/data/telecoupling_literature.bib"


def extract_keywords_from_markdown(text: str):
    """Return (list_of_keywords, source_label) or (None, reason)."""
    # Strategy 1: explicit heading like "# Keywords:" or "## Keywords"
    heading_match = re.search(
        r"^#+\s*Keywords?\s*:?\s*$", text, re.MULTILINE | re.IGNORECASE
    )
    if heading_match:
        rest = text[heading_match.end():]
        stop = re.search(r"\n#+\s|\n# |\n\n\n", rest)
        block = rest[: stop.start()] if stop else rest[:2000]
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        kws: list[str] = []
        for line in lines[:30]:
            if len(line) > 200:
                break
            if line.lower().startswith("abstract") or line.startswith("#"):
                break
            parts = re.split(r"[;,|]\s*", line)
            for p in parts:
                p = p.strip().strip(".").strip()
                if p:
                    kws.append(p)
        if kws:
            return kws, "heading"

    # Strategy 2: "Keywords:" on its own line followed by blank-separated paragraphs
    # (common in Elsevier/Wiley PDF conversions).
    para_header = re.search(
        r"^(?:Key\s*Words?|Keywords?)\s*:\s*$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )
    if para_header:
        rest = text[para_header.end():]
        # Skip leading blank lines
        rest = rest.lstrip("\n")
        paras = re.split(r"\n\s*\n", rest)
        kws: list[str] = []
        for p in paras[:40]:
            p = p.strip()
            if not p:
                continue
            if p.startswith("#"):
                break  # next heading
            # Stop conditions: too-long paragraph, looks like body text
            if len(p) > 150:
                break
            # Abstract section
            if re.match(r"^(Abstract|ABSTRACT|Introduction|Background)\b", p):
                break
            # Split on ; or , within a single paragraph
            if ";" in p or "," in p or "|" in p:
                parts = re.split(r"[;,|]\s*", p)
                for part in parts:
                    part = part.strip().strip(".").strip()
                    if part:
                        kws.append(part)
            else:
                kws.append(p.strip(".").strip())
        # Sanity: if we extracted >40 "keywords", something went wrong
        if kws and len(kws) <= 40:
            return kws, "paragraphs"

    # Strategy 3: inline "Keywords: X; Y; Z" or "Key Words: X; Y; Z"
    inline_match = re.search(
        r"^(?:Key\s*Words?|Keywords?)\s*:\s*(\S.+?)(?=\n\n|\n#)",
        text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if inline_match:
        val = re.sub(r"\s+", " ", inline_match.group(1).strip())
        if "|" in val:
            sep = "|"
        elif ";" in val:
            sep = ";"
        else:
            sep = ","
        kws = [p.strip().strip(".").strip() for p in val.split(sep)]
        kws = [k for k in kws if k and len(k) < 200]
        if kws:
            return kws, "inline"

    return None, "not_found"


def parse_bib(bib_text: str):
    """Return {key: {author, year, title}}."""
    out = {}
    starts = [m.start() for m in re.finditer(r"^@\w+\{", bib_text, re.MULTILINE)]
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(bib_text)
        chunk = bib_text[s:e]
        key_m = re.match(r"@\w+\{\s*([^,]+),", chunk)
        if not key_m:
            continue
        key = key_m.group(1).strip()
        author_m = re.search(r"author\s*=\s*\{([^}]+)\}", chunk, re.IGNORECASE)
        year_m = re.search(r"year\s*=\s*\{(\d{4})\}", chunk, re.IGNORECASE)
        title_m = re.search(r"title\s*=\s*\{([^}]+)\}", chunk, re.IGNORECASE)
        if not (author_m and year_m):
            continue
        first_author = author_m.group(1).split(" and ")[0].split(",")[0].strip()
        out[key] = {
            "author": first_author,
            "year": year_m.group(1),
            "title": title_m.group(1) if title_m else "",
        }
    return out


def _strip_diacritics(s: str) -> str:
    import unicodedata
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


# Manual overrides for filenames that the automated matcher can't resolve
# correctly (typically: ambiguous multi-author-same-year cases, compound
# surnames, or pre-existing bogus .bib keys).
MANUAL_OVERRIDES = {
    "Liu and Pan - 2025 - Unraveling the quantity and sustainability of cross-scale ecosystem service flows A meta-coupling f.md": "liu_unraveling_2025_a",
    "Liu et al. - 2025 - From Plate to Plow How Dietary Shifts Drive Telecoupled Cropland Erosion in China.md": "liu_plate_2025",
    # Filename credits "Li et al." but the actual paper's lead author is Zhao:
    "Li et al. - 2019 - Tightening ecological management facilitates green development in the Qilian Mountains.md": "zhao_tightening_2019",
    # Filename says "Reis" but BibTeX key uses full surname "dos Reis":
    "Reis et al. - 2020 - Understanding the Stickiness of Commodity Supply Chains Is Key to Improving Their Sustainability.md": "dosreis_understanding_2020",
}


def _surname_variants(raw: str) -> list[str]:
    """Return candidate surname stems from 'da Silva' / 'Moreno-Fernández' / etc.

    Produces forms like:
    - last-word:         'silva' / 'fernandez'
    - compound-joined:   'dasilva' / 'morenofernandez'
    - hyphen-split:      'moreno', 'fernandez'
    """
    raw = _strip_diacritics(raw).lower()
    raw = raw.replace(" et al.", "").strip()
    # Strip trailing punctuation
    raw = raw.rstrip(".,;")
    variants: list[str] = []
    # Remove leading "and" pieces (multi-author)
    first = raw.split(" and ")[0].strip()
    # If comma-separated, first part before comma is surname
    first = first.split(",")[0].strip()
    if not first:
        return variants
    # Tokenize
    tokens = [re.sub(r"[^a-z]", "", t) for t in first.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return variants
    # last-word
    variants.append(tokens[-1])
    # compound-joined (all tokens concatenated)
    if len(tokens) > 1:
        variants.append("".join(tokens))
    # hyphen-split pieces
    for t in first.replace(" ", "-").split("-"):
        t_clean = re.sub(r"[^a-z]", "", t)
        if t_clean and t_clean not in variants:
            variants.append(t_clean)
    return variants


def match_filename_to_bib(filename: str, bib_index: dict):
    """Resolve 'Agusdinata et al. - 2023 - Advancing...' → 'agusdinata_advancing_2023'."""
    # Check manual overrides first
    if filename in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[filename]

    m = re.match(r"^(.+?)\s*-\s*(\d{4})\s*-\s*(.+?)\.md$", filename, re.IGNORECASE)
    if not m:
        return None
    authors, year, title = m.group(1), m.group(2), m.group(3)

    # Build surname candidates
    surname_candidates = _surname_variants(authors)
    if not surname_candidates:
        return None

    # Title first word
    title_clean = re.sub(r"^(the|a|an)\s+", "", title.strip(), flags=re.IGNORECASE)
    title_words = [re.sub(r"[^a-z]", "", w.lower()) for w in title_clean.split()]
    title_words = [w for w in title_words if w]
    title_first = title_words[0] if title_words else ""

    # Phase 1: strict — match by (surname variant, year)
    candidates = []
    for key, meta in bib_index.items():
        if meta["year"] != year:
            continue
        key_parts = key.split("_")
        if len(key_parts) < 2:
            continue
        key_surname = re.sub(r"[^a-z]", "", key_parts[0].lower())
        if not key_surname:
            continue
        for surname in surname_candidates:
            if not surname:
                continue
            if key_surname == surname:
                candidates.append((key, meta, "exact"))
                break
            if len(surname) >= 4 and len(key_surname) >= 4 and key_surname[:4] == surname[:4]:
                candidates.append((key, meta, "prefix"))
                break

    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for key, meta, reason in candidates:
        if key not in seen:
            seen.add(key)
            uniq.append((key, meta, reason))
    candidates = uniq

    if len(candidates) == 1:
        return candidates[0][0]
    if len(candidates) > 1:
        # Disambiguate by title first-word
        for key, _, _ in candidates:
            key_parts = key.split("_")
            if len(key_parts) >= 2:
                key_firstword = re.sub(r"[^a-z]", "", key_parts[1].lower())
                if title_first and key_firstword:
                    if (
                        title_first.startswith(key_firstword[:4])
                        or key_firstword.startswith(title_first[:4])
                    ):
                        return key
        return candidates[0][0]

    # Phase 2: constrained fallback — only for .bib keys with the bogus
    # "al_" prefix (from a historical parser bug), where the surname part
    # is literally "al". Match by (year + title first-word).
    if title_first and len(title_first) >= 4:
        for key, meta in bib_index.items():
            if meta["year"] != year:
                continue
            key_parts = key.split("_")
            if len(key_parts) < 2:
                continue
            if key_parts[0].lower() != "al":
                continue  # only fire for the bogus-prefix case
            key_firstword = re.sub(r"[^a-z]", "", key_parts[1].lower())
            if not key_firstword:
                continue
            if key_firstword[:5] == title_first[:5]:
                return key
    return None


def main():
    with open(BIB_PATH, encoding="utf-8") as f:
        bib_text = f.read()
    bib_index = parse_bib(bib_text)
    print(f"BibTeX entries: {len(bib_index)}")

    md_files = sorted(glob.glob(os.path.join(MD_ROOT, "**", "*.md"), recursive=True))
    print(f"Markdown files: {len(md_files)}")
    print()

    stats = {"heading": 0, "paragraphs": 0, "inline": 0, "alt-caps": 0, "not_found": 0,
             "matched_to_bib": 0, "unmatched": 0}
    results = []
    unmatched_files = []
    bib_keys_seen = set()

    for path in md_files:
        filename = os.path.basename(path)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception:
            continue

        kws, source = extract_keywords_from_markdown(text)
        stats[source] += 1

        bib_key = match_filename_to_bib(filename, bib_index)
        if bib_key:
            stats["matched_to_bib"] += 1
            bib_keys_seen.add(bib_key)
        else:
            stats["unmatched"] += 1
            unmatched_files.append(filename)

        results.append((filename, bib_key, source, kws))

    # Save CSV with title-similarity flag to catch false positives.
    # Put it in the project tmp/ for easy access (gitignored).
    csv_dir = "tmp"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "keyword_extraction.csv")

    def title_similarity(fname: str, bib_title: str) -> float:
        """Crude word-overlap ratio between filename title fragment and bib title."""
        fm = re.match(r"^.+?-\s*\d{4}\s*-\s*(.+?)\.md$", fname)
        if not fm:
            return 0.0
        ft = set(re.findall(r"[a-z]{4,}", fm.group(1).lower()))
        bt = set(re.findall(r"[a-z]{4,}", (bib_title or "").lower()))
        if not ft or not bt:
            return 0.0
        return len(ft & bt) / max(len(ft), 1)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "bib_key", "title_overlap", "source",
                    "num_keywords", "keywords"])
        for filename, bib_key, source, kws in results:
            kw_str = "; ".join(kws) if kws else ""
            overlap = 0.0
            if bib_key and bib_key in bib_index:
                overlap = title_similarity(filename, bib_index[bib_key]["title"])
            w.writerow([filename, bib_key or "", f"{overlap:.2f}",
                        source, len(kws) if kws else 0, kw_str])

    # Compute usable extractions: has keywords AND matched to .bib
    usable = [(f, k, s, kws) for f, k, s, kws in results if kws and k]
    # Low-confidence matches: overlap < 0.2
    low_conf = []
    for f, k, _, kws in results:
        if kws and k and k in bib_index:
            fm = re.match(r"^.+?-\s*\d{4}\s*-\s*(.+?)\.md$", f)
            if fm:
                ft = set(re.findall(r"[a-z]{4,}", fm.group(1).lower()))
                bt = set(re.findall(r"[a-z]{4,}", bib_index[k]["title"].lower()))
                ov = len(ft & bt) / max(len(ft), 1) if ft else 0
                if ov < 0.2:
                    low_conf.append((f, k, ov))

    print("=== Extraction summary ===")
    print(f"  Keywords via heading:        {stats['heading']}")
    print(f"  Keywords via paragraphs:     {stats['paragraphs']}")
    print(f"  Keywords via inline:         {stats['inline']}")
    print(f"  Keywords via caps:           {stats['alt-caps']}")
    print(f"  Keywords NOT found:          {stats['not_found']}")
    print(f"  Matched to .bib key:         {stats['matched_to_bib']}")
    print(f"  Unmatched:                   {stats['unmatched']}")
    print()
    print(f"  USABLE (kw AND matched):     {len(usable)}")
    print(f"  Low-confidence matches:      {len(low_conf)}")
    print()
    print(f"CSV: {csv_path}")
    print()
    if low_conf:
        print("=== Low-confidence matches (title overlap < 0.2) ===")
        for f, k, ov in low_conf[:10]:
            print(f"  [ov={ov:.2f}] {f[:70]} -> {k}")
        if len(low_conf) > 10:
            print(f"  ... and {len(low_conf) - 10} more")
        print()

    # Bib entries with NO matching markdown
    missing_md = [k for k in bib_index if k not in bib_keys_seen]
    print(f"=== {len(missing_md)} .bib entries with no matching markdown ===")
    for k in missing_md[:20]:
        meta = bib_index[k]
        print(f"  {k}  ({meta['author']} {meta['year']})")
    if len(missing_md) > 20:
        print(f"  ... and {len(missing_md) - 20} more")
    print()

    # Sample extractions
    print("=== Sample extractions (first 5) ===")
    for filename, bib_key, source, kws in results[:5]:
        kw_preview = "; ".join(kws[:8]) if kws else "(none)"
        print(f"  [{source}] {filename[:90]}")
        print(f"     -> {bib_key or 'UNMATCHED'}")
        print(f"     -> keywords: {kw_preview}")
    print()

    no_kw = [(f, k) for f, k, s, kws in results if not kws]
    print(f"=== {len(no_kw)} papers where NO keywords found ===")
    for filename, bib_key in no_kw[:20]:
        print(f"  {filename[:90]} -> {bib_key or 'UNMATCHED'}")
    if len(no_kw) > 20:
        print(f"  ... and {len(no_kw) - 20} more")
    print()

    if unmatched_files:
        print(f"=== {len(unmatched_files)} files not matched to any .bib entry ===")
        for f in unmatched_files[:20]:
            print(f"  {f[:90]}")
        if len(unmatched_files) > 20:
            print(f"  ... and {len(unmatched_files) - 20} more")


if __name__ == "__main__":
    main()
