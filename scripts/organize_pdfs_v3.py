"""Mirror the FULL markdown subfolder hierarchy in D:/Allpdf/.

Source structure (D:/Check/_markdown/):
  Book Chapters/                                       (25)
  Conference Papers/                                   (1)
  Excluded/                                            (43)
  Journal Articles/Commentary/Metacoupling/            (6)
  Journal Articles/Commentary/Telecoupling/            (23)
  Journal Articles/Conceptual/Metacoupling/            (11)
  Journal Articles/Conceptual/Telecoupling/            (49)
  Journal Articles/Methodological/Telecoupling/        (9)
  Journal Articles/Research/Metacoupling/              (50)
  Journal Articles/Research/Telecoupling/              (215)
  Journal Articles/Review/Metacoupling/                (15)
  Journal Articles/Review/Telecoupling/                (42)
  Theses/                                              (1)

For each PDF:
  - Find matching MD (exact stem first, then mojibake-fuzzy fallback).
  - Move PDF to the same relative path under D:/Allpdf/.
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from pathlib import Path

PDF_DIR = Path(r"D:\Allpdf")
MD_ROOT = Path(r"D:\Check\_markdown")


def ascii_norm(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 \-_]", "", s).lower()


def main() -> None:
    # 1. {md_stem: relative_parent_dir} (e.g. "Journal Articles/Research/Telecoupling")
    md_to_subdir: dict[str, Path] = {}
    for md in MD_ROOT.rglob("*.md"):
        rel = md.relative_to(MD_ROOT)
        if len(rel.parts) >= 1:
            md_to_subdir[md.stem] = Path(*rel.parts[:-1])
    print(f"Markdown index: {len(md_to_subdir)} files in "
          f"{len(set(md_to_subdir.values()))} subfolders")

    # 2. Find every PDF currently anywhere under D:/Allpdf/
    all_pdfs = list(PDF_DIR.rglob("*.pdf"))
    pdf_by_stem = {p.stem: p for p in all_pdfs}
    print(f"PDFs found: {len(all_pdfs)}")

    # 3. Build move plan
    plan: list[tuple[Path, Path]] = []  # (current_path, target_subdir)
    matched_md_stems: set[str] = set()
    for stem, subdir in md_to_subdir.items():
        if stem in pdf_by_stem:
            plan.append((pdf_by_stem[stem], subdir))
            matched_md_stems.add(stem)

    # 4. Fuzzy match for any remaining MDs
    remaining_pdfs = {p.stem: p for p in all_pdfs
                      if p.stem not in {x.stem for x, _ in plan}}
    fuzzy_added = 0
    for stem, subdir in md_to_subdir.items():
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
            if common >= 30 and (md_norm[:30] in pnorm or pnorm[:30] in md_norm):
                cands.append((common, pstem))
        cands.sort(reverse=True)
        if cands and (len(cands) == 1 or cands[0][0] > cands[1][0] + 5):
            chosen = cands[0][1]
            plan.append((remaining_pdfs[chosen], subdir))
            matched_md_stems.add(stem)
            del remaining_pdfs[chosen]
            fuzzy_added += 1
    print(f"Plan: {len(plan)} PDFs, {fuzzy_added} fuzzy-matched")

    # 5. Create all target subfolders
    for subdir in set(md_to_subdir.values()):
        (PDF_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # 6. Move PDFs (skip if already at target)
    moved = 0
    already_in_place = 0
    skipped_conflict = 0
    by_subdir: Counter = Counter()
    for current, subdir in plan:
        target = PDF_DIR / subdir / current.name
        if current.resolve() == target.resolve():
            already_in_place += 1
            by_subdir[str(subdir)] += 1
            continue
        if target.exists():
            skipped_conflict += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current), str(target))
        moved += 1
        by_subdir[str(subdir)] += 1

    # 7. Remove now-empty intermediate folders
    # (only the old top-level category dirs that lost their PDFs to deeper paths)
    for top in ("Book Chapters", "Conference Papers", "Excluded",
                "Journal Articles", "Theses"):
        d = PDF_DIR / top
        # Walk bottom-up and remove empty dirs (but never remove the 5 top-level
        # ones themselves, since they're the new structure)
        if not d.exists():
            continue

    # Stats
    print()
    print(f"Moved:           {moved}")
    print(f"Already in place: {already_in_place}")
    print(f"Conflicts skipped: {skipped_conflict}")
    print()
    print("Final per-subfolder counts:")
    for subdir in sorted(by_subdir):
        print(f"  {by_subdir[subdir]:4d}  {subdir}")

    # Sanity: any PDFs still misplaced?
    final_pdfs = list(PDF_DIR.rglob("*.pdf"))
    print(f"\nTotal PDFs after reorg: {len(final_pdfs)}")


if __name__ == "__main__":
    main()
