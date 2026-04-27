"""Re-organize D:/Allpdf/ into the 5 top-level categories from
D:/Check/_markdown/:  Book Chapters, Conference Papers, Excluded,
Journal Articles, Theses.

Steps:
  1. Build {pdf_basename: top_category} from the full markdown tree.
  2. Consolidate any PDFs currently in D:/Allpdf/Telecoupling/ or
     D:/Allpdf/Metacoupling/ (left over from the previous run) back
     to D:/Allpdf/ root.
  3. Remove those obsolete subfolders if empty.
  4. Move each PDF into D:/Allpdf/<top_category>/.
  5. Fuzzy-match PDFs whose Windows-saved name is mojibake of the MD
     proper-Unicode name (the same 5 cases as before).
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

PDF_DIR = Path(r"D:\Allpdf")
MD_ROOT = Path(r"D:\Check\_markdown")

OBSOLETE_SUBFOLDERS = ("Telecoupling", "Metacoupling")
TOP_CATEGORIES = (
    "Book Chapters", "Conference Papers", "Excluded",
    "Journal Articles", "Theses",
)


def ascii_norm(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 \-_]", "", s).lower()


def main() -> None:
    # 1. {md_stem: top_category}
    md_map: dict[str, str] = {}
    for md in MD_ROOT.rglob("*.md"):
        rel = md.relative_to(MD_ROOT)
        if len(rel.parts) >= 1:
            md_map[md.stem] = rel.parts[0]
    print(f"Markdown index: {len(md_map)} files across "
          f"{len(set(md_map.values()))} top-level categories")

    # 2. Consolidate any PDFs from obsolete subfolders back to root
    moved_back = 0
    for sub in OBSOLETE_SUBFOLDERS:
        d = PDF_DIR / sub
        if not d.exists():
            continue
        for pdf in list(d.glob("*.pdf")):
            target = PDF_DIR / pdf.name
            if target.exists():
                continue
            shutil.move(str(pdf), str(target))
            moved_back += 1
    print(f"Consolidated {moved_back} PDFs back from obsolete subfolders")

    # 3. Remove now-empty obsolete subfolders
    for sub in OBSOLETE_SUBFOLDERS:
        d = PDF_DIR / sub
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            print(f"Removed empty {d}")

    # 4. Build PDF->category plan
    pdfs = list(PDF_DIR.glob("*.pdf"))
    pdf_by_stem = {p.stem: p for p in pdfs}
    plan: list[tuple[Path, str]] = []
    matched_md_stems: set[str] = set()
    for stem, cat in md_map.items():
        if stem in pdf_by_stem:
            plan.append((pdf_by_stem[stem], cat))
            matched_md_stems.add(stem)

    # 5. Fuzzy match for any remaining MD entries (mojibake PDFs)
    remaining_pdfs = set(pdf_by_stem) - {p.stem for p, _ in plan}
    fuzzy_added = 0
    for stem, cat in md_map.items():
        if stem in matched_md_stems:
            continue
        md_norm = ascii_norm(stem)
        cands = []
        for pstem in remaining_pdfs:
            pnorm = ascii_norm(pstem)
            common = 0
            for a, b in zip(md_norm, pnorm):
                if a == b:
                    common += 1
                else:
                    break
            if common >= 30 and md_norm[:30] in pnorm:
                cands.append((common, pstem))
        cands.sort(reverse=True)
        if cands and (len(cands) == 1 or cands[0][0] > cands[1][0] + 5):
            chosen = cands[0][1]
            plan.append((pdf_by_stem[chosen], cat))
            matched_md_stems.add(stem)
            remaining_pdfs.discard(chosen)
            fuzzy_added += 1
    print(f"Plan: {len(plan)} PDFs to move (incl. {fuzzy_added} fuzzy-matched)")

    # 6. Create the 5 top-level folders
    for cat in TOP_CATEGORIES:
        (PDF_DIR / cat).mkdir(exist_ok=True)

    # 7. Execute moves
    moved = 0
    skipped = []
    by_cat = Counter()
    for pdf, cat in plan:
        target = PDF_DIR / cat / pdf.name
        if target.exists():
            skipped.append(pdf.name)
            continue
        shutil.move(str(pdf), str(target))
        moved += 1
        by_cat[cat] += 1

    # 8. Anything still at root?
    still_root = list(PDF_DIR.glob("*.pdf"))

    print()
    print(f"Moved: {moved}")
    for cat, n in by_cat.most_common():
        print(f"  -> D:\\Allpdf\\{cat}\\: {n}")
    if skipped:
        print(f"Skipped (target existed): {len(skipped)}")
    print(f"PDFs still at D:\\Allpdf\\ root: {len(still_root)}")
    if still_root:
        for p in still_root[:10]:
            print(f"  {p.name[:120].encode('ascii','replace').decode('ascii')}")


if __name__ == "__main__":
    main()
