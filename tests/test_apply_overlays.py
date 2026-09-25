"""Tests for the Stage 3-4 engine and the one-command regeneration.

The engine (scripts/apply_overlays.py) applies Stages 3 and 4 on top of the
geometry build's Stages 1-2: the water-only classification from one file
(water_classification_pairs.csv), then the edge corrections (the land-gap
borders added, the denylisted contacts removed).  Its contract: running on
already-processed shipped data is a byte-stable no-op, a fresh build's output
reproduces the shipped files, and its inputs are exactly the reviewed files.
"""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
DATA = REPO / "src" / "metacouplingllm" / "data"

OUTPUT_FILES = [
    "pericoupled_adm1_edge_list.csv",
    "water_separated_pairs.csv",
    "PeriTelecoupling_clean.csv",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy_data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    for f in OUTPUT_FILES:
        shutil.copy(DATA / f, d / f)
    for f in _load("apply_overlays").INPUT_FILES:
        shutil.copy(DATA / f, d / f)
    return d


class TestApplyOverlays:
    def test_noop_on_shipped_data(self, tmp_path):
        """Running the engine on shipped data must change zero bytes."""
        d = _copy_data(tmp_path)
        before = {f: (d / f).read_bytes() for f in OUTPUT_FILES}
        mod = _load("apply_overlays")
        assert mod.main(["--data-dir", str(d)]) == 0
        for f in OUTPUT_FILES:
            assert (d / f).read_bytes() == before[f], f"{f} changed bytes"

    def test_noop_on_lf_normalized_data(self, tmp_path):
        """CI checks out with LF line endings (no autocrlf): still a no-op."""
        d = _copy_data(tmp_path)
        for f in OUTPUT_FILES:
            p = d / f
            p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        before = {f: (d / f).read_bytes() for f in OUTPUT_FILES}
        mod = _load("apply_overlays")
        assert mod.main(["--data-dir", str(d)]) == 0
        for f in OUTPUT_FILES:
            assert (d / f).read_bytes() == before[f], f"{f} changed bytes (LF)"

    def test_check_mode_reports_clean(self, tmp_path):
        d = _copy_data(tmp_path)
        mod = _load("apply_overlays")
        assert mod.main(["--data-dir", str(d), "--check"]) == 0

    def test_inputs_are_exactly_the_reviewed_files(self):
        """The engine reads the water file (Stage 3), the land-gap file and the
        denylist (Stage 4); a new *_overlay_pairs.csv it does not read, or a
        retired store left behind, must fail here."""
        mod = _load("apply_overlays")
        assert set(mod.INPUT_FILES) == {"water_classification_pairs.csv",
                                        "land_gap_overlay_pairs.csv", "denylist_pairs.csv"}
        assert all((DATA / f).exists() for f in mod.INPUT_FILES)
        overlays = {f.name for f in DATA.glob("*_overlay_pairs.csv")}
        overlays.discard("disputed_overlay_pairs.csv")  # written by the geometry build (Stage 2)
        assert overlays <= set(mod.INPUT_FILES)
        for retired in ("rescreen_gap_overlay_pairs.csv", "rescreen_water_overlay_pairs.csv"):
            assert not (DATA / retired).exists(), retired
        assert not (REPO / "build_data" / "bridge_classified_authoritative.csv").exists()

    def test_fresh_build_reproduces_the_shipped_files(self, tmp_path):
        """What a fresh --full build hands the engine: Stages 1-2 only (no water
        table, the 24 + 4 added edges absent, the five denylisted contacts
        present).  Stages 3-4 must reproduce the shipped files byte for byte."""
        d = _copy_data(tmp_path)
        mod = _load("apply_overlays")
        added = {frozenset({r["code_a"].strip(), r["code_b"].strip()})
                 for f in ("water_classification_pairs.csv", "land_gap_overlay_pairs.csv")
                 for r in _csv.DictReader(open(DATA / f, newline="", encoding="utf-8-sig"))
                 if f.startswith("land_gap") or r["adds_edge"] == "True"}
        edge = d / "pericoupled_adm1_edge_list.csv"
        eol = b"\r\n" if b"\r\n" in edge.read_bytes()[:4096] else b"\n"
        lines = edge.read_bytes().split(eol)
        head, body = lines[0], [x for x in lines[1:] if x]
        keep = [x for x in body
                if frozenset(x.decode("utf-8").split(",")[i] for i in (0, 5)) not in added]
        iso_meta = {}  # the engine takes a country's name and region from the edge list
        for r in _csv.DictReader(open(DATA / "pericoupled_adm1_edge_list.csv", newline="",
                                      encoding="utf-8-sig")):
            iso_meta.setdefault(r["ISO_A3_A"], (r["country_A"], r["WB_region_A"]))
            iso_meta.setdefault(r["ISO_A3_B"], (r["country_B"], r["WB_region_B"]))
        contacts = []
        for r in _csv.DictReader(open(DATA / "denylist_pairs.csv", newline="", encoding="utf-8-sig")):
            (ca, ra), (cb, rb) = iso_meta[r["iso_a"]], iso_meta[r["iso_b"]]
            contacts.append(",".join([r["code_a"], r["name_a"], ca, r["iso_a"], ra,
                                      r["code_b"], r["name_b"], cb, r["iso_b"], rb,
                                      str(r["iso_a"] != r["iso_b"]), "1.0", "True", "False"]).encode("utf-8"))
        assert len(keep) == 8461 - 28 and len(contacts) == 5
        edge.write_bytes(eol.join([head, *keep[:100], *contacts, *keep[100:]]) + eol)
        (d / "water_separated_pairs.csv").unlink()
        assert mod.main(["--data-dir", str(d)]) == 0
        for f in OUTPUT_FILES:
            assert (d / f).read_bytes() == (DATA / f).read_bytes(), f"{f} differs from the shipped file"


class TestBuildAll:
    def test_refresh_exits_zero(self, tmp_path):
        """The one-command refresh (engine + verify_counts) passes on shipped data."""
        d = _copy_data(tmp_path)
        r = subprocess.run(
            [sys.executable, "-X", "utf8", str(SCRIPTS / "build_all.py"),
             "--data-dir", str(d)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "build_all: OK" in r.stdout

    def test_verify_counts_catches_drift(self, tmp_path):
        """Dropping one edge row must fail verification with a named count."""
        d = _copy_data(tmp_path)
        edge = d / "pericoupled_adm1_edge_list.csv"
        lines = edge.read_bytes().splitlines(keepends=True)
        edge.write_bytes(b"".join(lines[:-1]))  # drop last data row
        mod = _load("build_all")
        fails = mod.verify_counts(d)
        assert any("adm1_edges" in f for f in fails)


# ---------------------------------------------------------------------------
# Structured provenance columns (`adjudication`, `verification_tier`)
# ---------------------------------------------------------------------------
#
# Added 2026-07-25.  The two columns are deliberately ORTHOGONAL:
#   `adjudication`      = the PROCESS class that produced the verdict
#   `verification_tier` = the EVIDENCE strength, with tier B pinned to the
#                         preregistered validation study's measured frame
# Widening tier B would falsify PROVENANCE.md's "98.7% precision" sentence with
# no visible failure, which is what test_tier_b_is_exactly_the_study_frame stops.

import csv as _csv
import re as _re
from collections import Counter

WATER_CSV = DATA / "water_separated_pairs.csv"
WATER_FILE = DATA / "water_classification_pairs.csv"
WATER_HEADER = ["level", "code_a", "code_b", "has_bridge", "water_type",
                "water_body", "note", "adjudication", "verification_tier"]


def _water_rows():
    with open(WATER_CSV, newline="", encoding="utf-8-sig") as fh:
        return list(_csv.DictReader(fh))


def test_water_csv_schema_and_notes():
    """The shipped water table's exact 9-column header, and its `note` column:
    the engine writes one note per instrument class (779 rows on a shared edge,
    24 between non-touching units) and `adm1-rollup` for the 26 ADM0 rows."""
    mod = _load("apply_overlays")
    with open(WATER_CSV, newline="", encoding="utf-8-sig") as fh:
        header = next(_csv.reader(fh))
        widths = {len(r) for r in _csv.reader(fh) if r}
    assert header == WATER_HEADER == mod.WATER_HEADER, f"water CSV header drifted: {header}"
    assert widths == {9}, f"ragged water CSV -- row widths {sorted(widths)}"
    notes = Counter(r["note"] for r in _water_rows())
    assert notes == Counter({mod.NOTE_ON_EDGE: 779, mod.NOTE_NON_TOUCHING: 24, "adm1-rollup": 26}), notes


def test_water_file_is_the_water_table():
    """Stage 3's one file carries every ADM1 water row, in the table's order:
    the same pairs, flags, types, bodies and provenance classes; the 24
    `adds_edge` rows are the non-touching borders, with a corridor length."""
    table = [r for r in _water_rows() if r["level"] == "adm1"]
    rows = list(_csv.DictReader(open(WATER_FILE, newline="", encoding="utf-8-sig")))
    assert len(rows) == len(table) == 803
    cols = ("code_a", "code_b", "has_bridge", "water_type", "water_body", "adjudication",
            "verification_tier")
    for w, t in zip(rows, table):  # the engine strips each value it writes
        assert tuple(w[c].strip() for c in cols) == tuple(t[c] for c in cols), (w, t)
    adds = [w for w in rows if w["adds_edge"] == "True"]
    assert len(adds) == 24 and all(float(w["border_km"]) > 0 for w in adds)
    assert all(w["adds_edge"] == "False" and not w["border_km"] for w in rows if w not in adds)


def test_denylist_is_exactly_the_reviewed_pairs():
    """Stage 4 removes exactly the five maintainer-decided non-adjacent contacts
    (docs/FUTURE_EDGE_AUDITS.md #7, #8, #11-#13); none ships as an edge or as a
    water row, and each carries its evidence and ruling."""
    deny = list(_csv.DictReader(open(DATA / "denylist_pairs.csv", newline="", encoding="utf-8-sig")))
    keys = {frozenset({r["code_a"], r["code_b"]}) for r in deny}
    assert keys == {frozenset({"LBR006", "LBR014"}), frozenset({"VEN001", "VEN003"}),
                    frozenset({"CAN003", "CAN006"}), frozenset({"COD009", "UGA102"}),
                    frozenset({"TZA016", "UGA040"})}
    assert all(r["evidence"].strip() and r["ruling"].strip() for r in deny)
    edges = {frozenset({r["ADM1_code_A"], r["ADM1_code_B"]})
             for r in _csv.DictReader(open(DATA / "pericoupled_adm1_edge_list.csv", newline="",
                                           encoding="utf-8-sig"))}
    assert not keys & edges
    assert not keys & {frozenset({r["code_a"], r["code_b"]}) for r in _water_rows()}


def test_every_adm1_row_has_provenance_and_adm0_has_none():
    """Both columns populated for every adm1 row; both blank for every adm0 row.

    ADM0 roll-ups are derived arithmetic (water-only iff every ADM1 crossing is;
    bridged iff any is), never adjudicated -- giving them a tier would imply a
    verification that never happened.
    """
    rows = _water_rows()
    adm1 = [r for r in rows if r["level"] == "adm1"]
    adm0 = [r for r in rows if r["level"] == "adm0"]
    missing = [f"{r['code_a']}<->{r['code_b']}" for r in adm1
               if not r["adjudication"].strip() or not r["verification_tier"].strip()]
    assert not missing, f"adm1 rows missing provenance: {missing[:10]}"
    filled = [f"{r['code_a']}<->{r['code_b']}" for r in adm0
              if r["adjudication"].strip() or r["verification_tier"].strip()]
    assert not filled, f"adm0 roll-up rows must carry no provenance: {filled}"


def test_adjudication_is_uniformly_cross_vendor():
    """wu1 + wu2's headline claim, asserted against the shipped data.

    Every shipped water-only row has been through the same cross-vendor
    two-pass: ru1 (2026-07-21) the river class, wu1 (2026-07-25) the 94 rows
    that had never been through it, wu2 (2026-07-25) the last 4 that had been
    settled by identity audit or the validation study instead.
    """
    adm1 = [r for r in _water_rows() if r["level"] == "adm1"]
    values = {r["adjudication"].strip() for r in adm1}
    assert values == {"cross-vendor"}, f"adjudication is not uniform: {sorted(values)}"
    assert len(adm1) == 803, len(adm1)


def test_tier_b_is_exactly_the_validation_study_frame():
    """Tier B == the 238 rows the preregistered validation study measured.

    PROVENANCE.md states tier-B precision 98.7% (95% CI [92.9%, 99.97%]) from
    that study.  If a later campaign widens tier B, the sentence silently becomes
    a claim about rows the study never sampled.  Pin the cardinality AND the
    predicate.

    The predicate is the hyphenated-or-spaced phrase "dual-AI verified", NOT the
    bare token "dual-AI": one non-touching row's source reads "overrules dual-AI
    corner verdict" -- a dual-AI verdict REJECTED by maintainer map ruling. A
    bare-token match returns 239 and quietly corrupts the frame.
    """
    dual = _re.compile(r"dual-AI[- ]verified")
    adm1 = [r for r in _water_rows() if r["level"] == "adm1"]
    tier_b = {f"{r['code_a']}<->{r['code_b']}" for r in adm1
              if r["verification_tier"].strip() == "B"}
    assert len(tier_b) == 238, (
        f"tier B has {len(tier_b)} rows, must be exactly the 238-row study frame "
        "(docs/VALIDATION_SAMPLING_PLAN.md); PROVENANCE's 98.7% precision claim "
        "is scoped to it")
    with open(WATER_FILE, newline="", encoding="utf-8-sig") as fh:
        from_source = {f"{r['code_a']}<->{r['code_b']}" for r in _csv.DictReader(fh)
                       if dual.search(r.get("source", ""))}
    assert tier_b == from_source, (
        "tier-B membership does not match the rows whose `source` in the water "
        "file records dual-AI verification")


def test_water_file_carries_the_provenance_columns():
    """The engine composes the water table from the water file on every run and
    on a fresh ``--full`` build; a missing provenance column there would blank
    provenance for every water row."""
    with open(WATER_FILE, newline="", encoding="utf-8-sig") as fh:
        rows = list(_csv.DictReader(fh))
    assert {"adjudication", "verification_tier", "source", "adds_edge"} <= set(rows[0])
    assert all(r["adjudication"].strip() and r["verification_tier"].strip() for r in rows)
