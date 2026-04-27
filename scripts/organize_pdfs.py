"""Mirror the markdown folder structure (Telecoupling / Metacoupling)
into D:/Allpdf/ by moving each matched PDF into the right subfolder.

Match strategy:
  1. Exact basename match (260 PDFs).
  2. ASCII-prefix fuzzy match for 5 PDFs whose Windows-saved names have
     mojibake (Latin-1-double-encoded UTF-8) while the markdown filenames
     have proper Unicode.

Unmatched PDFs (~225) stay at D:/Allpdf/ root.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

PDF_DIR = Path(r"D:\Allpdf")
MD_DIR = Path(r"D:\Check\_markdown\Journal Articles\Research")


def ascii_norm(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 \-_]", "", s).lower()


def main() -> None:
    # Build {md_stem: subfolder_name}
    md_map: dict[str, str] = {}
    for md in MD_DIR.rglob("*.md"):
        rel = md.relative_to(MD_DIR)
        if len(rel.parts) >= 2:
            md_map[md.stem] = rel.parts[0]

    pdfs = list(PDF_DIR.glob("*.pdf"))
    pdf_by_stem = {p.stem: p for p in pdfs}

    # 1. Exact matches
    plan: list[tuple[Path, str]] = []  # (pdf_path, subfolder)
    matched_md_stems = set()
    for stem, sub in md_map.items():
        if stem in pdf_by_stem:
            plan.append((pdf_by_stem[stem], sub))
            matched_md_stems.add(stem)

    # 2. Fuzzy match for remaining MDs (mojibake PDFs)
    remaining_pdf_stems = set(pdf_by_stem) - {p.stem for p, _ in plan}
    for stem, sub in md_map.items():
        if stem in matched_md_stems:
            continue
        md_norm = ascii_norm(stem)
        candidates: list[tuple[int, str]] = []
        for pstem in remaining_pdf_stems:
            pnorm = ascii_norm(pstem)
            common = 0
            for a, b in zip(md_norm, pnorm):
                if a == b:
                    common += 1
                else:
                    break
            if common >= 30 and md_norm[:30] in pnorm:
                candidates.append((common, pstem))
        candidates.sort(reverse=True)
        if candidates and (
            len(candidates) == 1 or candidates[0][0] > candidates[1][0] + 5
        ):
            chosen = candidates[0][1]
            plan.append((pdf_by_stem[chosen], sub))
            matched_md_stems.add(stem)
            remaining_pdf_stems.discard(chosen)

    # Create target subfolders
    for sub in set(md_map.values()):
        (PDF_DIR / sub).mkdir(exist_ok=True)

    # Execute moves
    moved: list[tuple[str, str]] = []
    skipped_existing: list[str] = []
    for pdf, sub in plan:
        target = PDF_DIR / sub / pdf.name
        if target.exists():
            skipped_existing.append(pdf.name)
            continue
        shutil.move(str(pdf), str(target))
        moved.append((pdf.name, sub))

    # Report
    from collections import Counter
    moved_by_sub = Counter(sub for _, sub in moved)
    print(f"PDFs moved:                   {len(moved)}")
    for sub, n in moved_by_sub.most_common():
        print(f"  -> D:\\Allpdf\\{sub}: {n}")
    print(f"Skipped (target exists):      {len(skipped_existing)}")
    if skipped_existing:
        for name in skipped_existing[:5]:
            print(f"    {name}")

    # Recount what's still at the root
    still_root = list(PDF_DIR.glob("*.pdf"))
    print(f"PDFs remaining at D:\\Allpdf\\ root: {len(still_root)}")


if __name__ == "__main__":
    main()
