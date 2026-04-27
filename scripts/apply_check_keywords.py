"""Apply user's manual keywords from Check.xlsx (36 rows) to the .bib.

Rules:
  - Row keyword == "NA" (or empty)  -> DELETE keywords field if present.
  - Row has keywords                -> REPLACE existing or INSERT new.

Also restores two entries whose keywords got corrupted in the previous
integration run (filename-collision bug):
  - du_how_2022     (was overwritten with "How far..." kw from du_how_2022_a)
  - friis_land_2017 (was overwritten with "On the System..." kw from friis_system_2017)
"""
from __future__ import annotations

import re
from pathlib import Path

import openpyxl

BIB_PATH = Path("src/metacoupling/data/telecoupling_literature.bib")
XLSX_PATH = Path(
    r"D:/Onedrive/OneDrive - Michigan State University/Desktop/Check.xlsx"
)

# For these keys the user used commas (not semicolons) as separators.
COMMA_SPLITS = {
    "zhao_tightening_2019",
    "latthachack_agricultural_2023",
}

# Per-entry raw-string cleanups applied before tokenizing.
SPECIFIC_CLEANUPS = {
    # Final item in dou is a comma-list inside a semicolon-list; split it.
    "dou_spillover_2018": ("deforestation, conservation, development",
                           "deforestation; conservation; development"),
}

# Restore correct keywords for entries corrupted by the earlier collision bug.
COLLISION_FIXES = {
    "du_how_2022": (
        "Virtual water trade; Environmental impacts; Water stress; "
        "Trade scenarios; Metacoupling; Telecoupling"
    ),
    "friis_land_2017": (
        "case-study research; Chinese investments; distal flows; feedbacks; "
        "land systems; land-use change; Laos; qualitative research; telecoupling"
    ),
}


def load_check_xlsx() -> list[tuple[str, str]]:
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    sh = wb["Sheet1"]
    rows = []
    for i, r in enumerate(sh.iter_rows(values_only=True)):
        if i == 0:
            continue
        if not r or not r[0]:
            continue
        key = str(r[0]).strip()
        kw = str(r[1]).strip() if r[1] is not None else ""
        rows.append((key, kw))
    return rows


def normalize_kw(key: str, raw: str) -> str:
    if key in SPECIFIC_CLEANUPS:
        find, replace = SPECIFIC_CLEANUPS[key]
        raw = raw.replace(find, replace)
    sep = r"[;,]" if key in COMMA_SPLITS else r";"
    parts = [p.strip().rstrip(".").strip() for p in re.split(sep, raw)]
    parts = [p for p in parts if p]
    return "; ".join(parts)


def rewrite_entry(entry_text: str, new_kw: str | None) -> tuple[str, bool]:
    """Returns (new_entry, changed). new_kw=None means delete."""
    kw_re = re.compile(
        r"(?P<indent>\n\s+)keywords\s*=\s*\{[^}]*\},?",
        re.IGNORECASE,
    )
    m = kw_re.search(entry_text)

    if new_kw is None:
        if not m:
            return entry_text, False
        return entry_text[: m.start()] + entry_text[m.end():], True

    new_field = f"keywords = {{{new_kw}}},"
    if m:
        indent = m.group("indent")
        return (entry_text[: m.start()] + indent + new_field
                + entry_text[m.end():]), True

    # Insert: before the closing `\n}` of the entry.
    close = re.search(r"(\n\s*\})\s*$", entry_text)
    if not close:
        return entry_text, False
    prev = re.search(r"\n(\s+)\w+\s*=", entry_text)
    indent = prev.group(1) if prev else "  "
    before = entry_text[: close.start()]
    if not before.rstrip().endswith(","):
        before = before.rstrip() + ","
    after = entry_text[close.start():]
    return before + f"\n{indent}{new_field}" + after, True


def main():
    rows = load_check_xlsx()
    print(f"Loaded {len(rows)} rows from Check.xlsx")

    directives: dict[str, tuple[str, str | None]] = {}
    for key, raw in rows:
        if raw.upper() == "NA" or not raw:
            directives[key] = ("delete", None)
        else:
            directives[key] = ("replace", normalize_kw(key, raw))

    for key, kw in COLLISION_FIXES.items():
        if key in directives:
            print(f"WARN: {key} already has directive; skipping collision fix")
            continue
        directives[key] = ("replace", kw)

    print(f"Directives: "
          f"{sum(1 for _, (a, _) in directives.items() if a == 'replace')} replace, "
          f"{sum(1 for _, (a, _) in directives.items() if a == 'delete')} delete")

    bib_text = BIB_PATH.read_text(encoding="utf-8")
    starts = [m.start() for m in re.finditer(r"^@\w+\{", bib_text, re.MULTILINE)]
    out: list[str] = []
    cursor = 0
    stats = {"replaced": 0, "inserted": 0, "deleted": 0, "noop": 0}
    seen: set[str] = set()
    detail: list[tuple[str, str]] = []

    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(bib_text)
        chunk = bib_text[start:end]
        out.append(bib_text[cursor:start])

        km = re.match(r"@\w+\{\s*([^,]+)\s*,", chunk)
        if not km:
            out.append(chunk)
            cursor = end
            continue
        key = km.group(1).strip()
        seen.add(key)

        em = re.search(r"\n\}\s*(\n|\Z)", chunk)
        if em:
            entry_text = chunk[: em.start() + 2]
            trailing = chunk[em.start() + 2:]
        else:
            entry_text = chunk
            trailing = ""

        if key in directives:
            action, kw = directives[key]
            had_kw = bool(re.search(r"\bkeywords\s*=", entry_text, re.IGNORECASE))
            new_entry, changed = rewrite_entry(
                entry_text, kw if action == "replace" else None
            )
            if changed:
                if action == "delete":
                    stats["deleted"] += 1
                    detail.append((key, "DEL"))
                elif had_kw:
                    stats["replaced"] += 1
                    detail.append((key, "REP"))
                else:
                    stats["inserted"] += 1
                    detail.append((key, "INS"))
            else:
                stats["noop"] += 1
                detail.append((key, "noop"))
            out.append(new_entry + trailing)
        else:
            out.append(chunk)
        cursor = end

    out.append(bib_text[cursor:])
    new_bib = "".join(out)

    missing = set(directives) - seen
    if missing:
        print(f"WARN: {len(missing)} directive keys not found in .bib: "
              f"{sorted(missing)}")

    BIB_PATH.write_text(new_bib, encoding="utf-8", newline="\n")
    print(f"\n  REPLACED:  {stats['replaced']}")
    print(f"  INSERTED:  {stats['inserted']}")
    print(f"  DELETED:   {stats['deleted']}")
    print(f"  NOOP:      {stats['noop']}")

    print("\nDetail (action per directive key):")
    for k, act in detail:
        print(f"  {act:5s}  {k}")

    print(f"\nWrote {BIB_PATH}")


if __name__ == "__main__":
    main()
