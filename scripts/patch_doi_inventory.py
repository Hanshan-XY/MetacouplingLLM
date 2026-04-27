"""Apply user-provided DOIs (from D:/Onedrive/.../Desktop/doi.xlsx) to
tmp/doi_inventory.json so the next run of check_oa_status.py picks them up.

For the 17 PDFs that had no DOI:
  - 16 user-provided DOIs are inserted with doi_source='manual'.
  - 1 row marked 'NA' (Strecker 2023) is left as not_found.

Filename matching:
  - Exact match first.
  - ASCII-prefix fuzzy match for mojibake-named files (Strecker, Garnero).
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import openpyxl

# Reroute stdout through UTF-8 so mojibake chars don't crash the GBK console.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")

DOI_INVENTORY = Path("tmp/doi_inventory.json")
DOI_XLSX = Path(r"D:/Onedrive/OneDrive - Michigan State University/Desktop/doi.xlsx")


def ascii_norm(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9 \-_]", "", s).lower()


def normalize_doi(s: str) -> str:
    """Strip any URL prefix, return bare DOI like '10.xxxx/yyyy'."""
    s = s.strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    return s


def main() -> None:
    inventory = json.loads(DOI_INVENTORY.read_text(encoding="utf-8"))
    print(f"Inventory: {len(inventory)} entries")

    # Read user's xlsx
    wb = openpyxl.load_workbook(DOI_XLSX, read_only=True, data_only=True)
    sh = wb["Sheet1"]
    pairs: list[tuple[str, str]] = []
    for i, r in enumerate(sh.iter_rows(values_only=True)):
        if i == 0:
            continue
        if not r or not r[0]:
            continue
        fn = str(r[0]).strip()
        doi_raw = str(r[1]).strip() if r[1] else ""
        if not doi_raw or doi_raw.upper() == "NA":
            print(f"  skip (NA): {_safe(fn)[:80]}")
            continue
        pairs.append((fn, normalize_doi(doi_raw)))
    print(f"Manual DOI rows: {len(pairs)}")

    # Build a lookup from inventory: pdf_path_str -> entry
    # Match by basename
    inv_by_basename: dict[str, str] = {Path(p).name: p for p in inventory}
    inv_by_basename_norm: dict[str, str] = {
        ascii_norm(Path(p).name): p for p in inventory
    }

    updated = 0
    skipped = []
    for fn, doi in pairs:
        path_key = inv_by_basename.get(fn)
        if not path_key:
            # fuzzy
            target_norm = ascii_norm(fn)
            matches = [p for nb, p in inv_by_basename_norm.items()
                       if nb[:60] == target_norm[:60]]
            if len(matches) == 1:
                path_key = matches[0]
        if not path_key:
            skipped.append(fn)
            continue
        prev = inventory[path_key]
        inventory[path_key] = {
            "doi": doi,
            "doi_source": "manual",
            "bib_key": prev.get("bib_key", ""),
        }
        updated += 1
        print(f"  + {_safe(Path(path_key).name)[:80]}")
        print(f"      doi: {doi}  (was: source={prev.get('doi_source')!r}, doi={prev.get('doi')!r})")

    DOI_INVENTORY.write_text(json.dumps(inventory, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    print()
    print(f"Updated: {updated}")
    print(f"Skipped (no inventory match): {len(skipped)}")
    for f in skipped:
        print(f"  ? {_safe(f)[:100]}")


if __name__ == "__main__":
    main()
