"""Guard test: every CSV shipped in src/metacouplingllm/data/ is well-formed.

A field holding an unquoted comma is two fields to a CSV reader, and nothing
fails: ``csv.DictReader`` files the text after the comma under the ``None`` key
and hands back the named field cut short.  Four notes of
``sliver_corridor_relabel.csv`` (SOM007, KEN046, SOM006, UGA065) shipped that
way -- ``... (69.73->85.09 km, verified by live re-run 2026-09-22).`` read back
as ending at ``km`` -- unnoticed, because the build reads only the host, owner
and opening columns of that file.

The test parses every shipped CSV with the ``csv`` module in strict mode and
requires each row to have exactly its header's field count.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "src" / "metacouplingllm" / "data"


class TestDataCsvs:
    def test_every_row_has_the_headers_field_count(self):
        """Each row of each shipped CSV parses to exactly as many fields as its header."""
        csvs = sorted(DATA.rglob("*.csv"))
        assert csvs, f"no CSV found under {DATA}"
        bad = []
        for path in csvs:
            name = path.relative_to(DATA).as_posix()
            try:
                # utf-8-sig: adm1_aliases.csv starts with a byte-order mark
                with open(path, newline="", encoding="utf-8-sig") as fh:
                    reader = csv.reader(fh, strict=True)
                    header = next(reader)
                    bad += [f"{name} line {reader.line_num}: {len(row)} fields, header has {len(header)}"
                            for row in reader if len(row) != len(header)]
            except (csv.Error, UnicodeDecodeError) as exc:
                bad.append(f"{name}: {exc}")
        assert not bad, "\n".join(bad)
