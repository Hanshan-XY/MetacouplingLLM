"""Tests for the unified overlay engine and the one-command rebuild.

The engine (scripts/apply_overlays.py) applies the reviewed correction layer
(three overlay manifests) on top of the geometry build.  Its contract: running
on already-overlaid shipped data is a byte-stable no-op, and the registry
covers exactly the shipped overlay manifests.
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
    for f in DATA.glob("*_overlay_pairs.csv"):
        shutil.copy(f, d / f.name)
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

    def test_registry_covers_all_shipped_manifests(self):
        """A new *_overlay_pairs.csv without a registry entry must fail here."""
        mod = _load("apply_overlays")
        registry_manifests = {entry[1] for entry in mod.REGISTRY}
        shipped = {f.name for f in DATA.glob("*_overlay_pairs.csv")}
        shipped.discard("disputed_overlay_pairs.csv")  # applied by the build script
        assert registry_manifests == shipped


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

WATER_CSV = DATA / "water_separated_pairs.csv"
BRIDGE_CSV = REPO / "build_data" / "bridge_classified_authoritative.csv"
WATER_HEADER = ["level", "code_a", "code_b", "has_bridge", "water_type",
                "water_body", "note", "adjudication", "verification_tier"]


def _water_rows():
    with open(WATER_CSV, newline="", encoding="utf-8-sig") as fh:
        return list(_csv.DictReader(fh))


def test_water_csv_schema_and_note_position():
    """Exact 9-column header with `note` at index 6 -- a positional contract.

    ``apply_overlays.py`` decides whether an existing row belongs to the overlay
    being applied with a bare positional compare, ``adm1_rows[i][5] == note``
    (index 5 in the tuple; 6 in the CSV, which prepends ``level``).  Inserting a
    column before ``note`` makes that test never match, so every overlay silently
    stops claiming its rows -- no error, the manifests just go inert.  Pin it.
    """
    with open(WATER_CSV, newline="", encoding="utf-8-sig") as fh:
        header = next(_csv.reader(fh))
        widths = {len(r) for r in _csv.reader(fh) if r}
    assert header == WATER_HEADER, f"water CSV header drifted: {header}"
    assert header.index("note") == 6, "note must stay at CSV index 6"
    assert widths == {9}, f"ragged water CSV -- row widths {sorted(widths)}"


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
    assert len(adm1) == 751, len(adm1)


def test_tier_b_is_exactly_the_validation_study_frame():
    """Tier B == the 238 rows the preregistered validation study measured.

    PROVENANCE.md states tier-B precision 98.7% (95% CI [92.9%, 99.97%]) from
    that study.  If a later campaign widens tier B, the sentence silently becomes
    a claim about rows the study never sampled.  Pin the cardinality AND the
    predicate.

    The predicate is the hyphenated-or-spaced phrase "dual-AI verified", NOT the
    bare token "dual-AI": one rescreen-gap row's source reads "overrules dual-AI
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
    from_source = set()
    for name in ("rescreen_water_overlay_pairs.csv", "rescreen_gap_overlay_pairs.csv"):
        with open(DATA / name, newline="", encoding="utf-8-sig") as fh:
            for r in _csv.DictReader(fh):
                if dual.search(r.get("source", "")):
                    from_source.add(f"{r['code_a']}<->{r['code_b']}")
    assert tier_b == from_source, (
        "tier-B membership does not match the rows whose manifest `source` "
        "records dual-AI verification")


def test_all_provenance_bearing_files_carry_the_columns():
    """Every registry manifest with water rows, plus the bridge CSV.

    The bridge CSV matters because ``build_pericoupling_db``'s
    ``write_water_separated_manifest`` regenerates the water CSV *from* it on
    ``--full``; omitting it there would blank provenance for the 298 base rows on
    every rebuild and break the byte-identity claim in docs/REPRODUCING.md.
    """
    # hydro_water / hydro_lakes retired 2026-07-28 -- their rows (and columns)
    # now live in rescreen_water
    targets = ["rescreen_gap_overlay_pairs.csv", "rescreen_water_overlay_pairs.csv"]
    for name in targets:
        with open(DATA / name, newline="", encoding="utf-8-sig") as fh:
            cols = next(_csv.reader(fh))
        assert "adjudication" in cols and "verification_tier" in cols, name
    with open(BRIDGE_CSV, newline="", encoding="utf-8-sig") as fh:
        cols = next(_csv.reader(fh))
    assert "adjudication" in cols and "verification_tier" in cols, BRIDGE_CSV.name
