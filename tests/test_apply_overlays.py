"""Tests for the unified overlay engine and the one-command rebuild.

The engine (scripts/apply_overlays.py) applies the reviewed correction layer
(nine overlay manifests) on top of the geometry build.  Its contract: running
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
