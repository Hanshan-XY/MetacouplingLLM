r"""Regenerate or verify the pericoupling database with one command.

The shipped database is the output of four stages, run in order:

1-2. **Geometry build** (``scripts/build_pericoupling_db.py``) from the pinned
     World Bank Official Boundaries GeoPackages (2026-05-14 release; SHA-256
     pins below and in ``data/PROVENANCE.md``): Stage 1, exact-contact
     contiguity with the source-relabel and the unit merge; Stage 2, the de
     facto overlay across disputed areas.
3-4. **Reviewed inputs** (``scripts/apply_overlays.py``): Stage 3, the
     water-only classification (``data/water_classification_pairs.csv``);
     Stage 4, the edge corrections (``data/land_gap_overlay_pairs.csv``,
     ``data/denylist_pairs.csv``).

Step-by-step reproduction manual: ``docs/REPRODUCING.md``.

Modes:

* ``python scripts/build_all.py``
    Refresh: re-apply Stages 3 and 4 to the shipped data and verify every
    headline count.  On an untouched checkout this is a byte-stable
    no-op that exits 0 -- the day-to-day reproducibility check.

* ``python scripts/build_all.py --full --adm1-gpkg ... --adm0-gpkg ...
    --ocean-gpkg ... --ndlsa-gpkg ... [--out-dir ...]``
    Full regeneration: verify the GeoPackages' SHA-256 against the pins (hard
    error on mismatch -- a changed input is a data-change PR that must update
    the pins), run the geometry build (Stages 1-2), check that every
    denylisted contact is present in its output, apply Stages 3-4, verify
    counts, and report whether the regenerated files match the committed ones.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_DATA = REPO / "src" / "metacouplingllm" / "data"

# SHA-256 of the pinned inputs (World Bank Official Boundaries, 2026-05-14
# release), of the raw bytes.  Also recorded in data/PROVENANCE.md "Sources
# (pinned)".  The reviewed inputs of Stages 3 and 4 live in the data directory
# under version control and are checked by the byte-identity report instead.
PINNED_SHA256 = {
    "adm1_gpkg": "dbac29f4ecaabe6a9b3ecf50780e5e57a725c7f0eab3d2514ce54fb717b64b45",
    "adm0_gpkg": "97f0c8a0fa848b9a8414dbeb2e058fa37d59b13794ec232a87da000bdf4b117e",
    "ocean_gpkg": "c2b074fdd691f6d36ba4a89af2761a11b35dea4d4c8c4f186f6132f43c88d702",
    "ndlsa_gpkg": "159ef2d133d12491eb6ce2f0d0d1032083209b0cf7d28ddda774a503055d2fa4",
}

EXPECTED = {
    "adm1_edges": 8461, "adm1_regions": 3374, "adm1_countries": 196,
    "water_adm1": 803, "water_bridge": 407, "water_nobridge": 396,
    "water_adm0": 26,
    "adm1_moderate": 8065, "adm1_stringent": 7658,
    "adm0_pairs": 326, "adm0_moderate": 320, "adm0_stringent": 300,
}


def _sha256(path: Path) -> str:
    """SHA-256 of a pinned input (the GeoPackages are binary: raw bytes)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_counts(data_dir: Path) -> list[str]:
    """Check every headline count against EXPECTED; return failure messages."""
    fails: list[str] = []

    def check(name: str, got: int) -> None:
        want = EXPECTED[name]
        if got != want:
            fails.append(f"{name}: expected {want}, got {got}")

    # edge list
    pairs, codes, countries = set(), set(), set()
    pair_iso: dict[frozenset, tuple[str, str]] = {}
    with open(data_dir / "pericoupled_adm1_edge_list.csv", newline="",
              encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            a, b = r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()
            pair = frozenset({a, b})
            pairs.add(pair)
            codes.update((a, b))
            ia, ib = r["ISO_A3_A"].strip(), r["ISO_A3_B"].strip()
            countries.update((ia, ib))
            pair_iso[pair] = (ia, ib)
    check("adm1_edges", len(pairs))
    check("adm1_regions", len(codes))
    check("adm1_countries", len(countries))

    # water manifest
    water, bridged = set(), set()
    adm0_roll = []
    with open(data_dir / "water_separated_pairs.csv", newline="",
              encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            key = frozenset({r["code_a"].strip(), r["code_b"].strip()})
            if r["level"] == "adm1":
                water.add(key)
                if r["has_bridge"].strip() == "True":
                    bridged.add(key)
            else:
                adm0_roll.append((key, r["has_bridge"].strip() == "True"))
    check("water_adm1", len(water))
    check("water_bridge", len(bridged))
    check("water_nobridge", len(water - bridged))
    check("water_adm0", len(adm0_roll))
    if not water <= pairs:
        fails.append(f"water rows not in edge list: {sorted(map(sorted, water - pairs))}")

    # standard views
    check("adm1_moderate", len(pairs) - len(water - bridged))
    check("adm1_stringent", len(pairs) - len(water))

    # ADM0 matrix + views
    adm0_pairs = set()
    with open(data_dir / "PeriTelecoupling_clean.csv", newline="",
              encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["Intracoupling"].strip() == "1":
                adm0_pairs.add(frozenset({r["Sending"].strip(), r["Receiving"].strip()}))
    check("adm0_pairs", len(adm0_pairs))
    roll_nobridge = sum(1 for _key, br in adm0_roll if not br)
    check("adm0_moderate", len(adm0_pairs) - roll_nobridge)
    check("adm0_stringent", len(adm0_pairs) - len(adm0_roll))

    # ADM0 roll-up must be reproducible from the ADM1 layer
    by_country = collections.defaultdict(list)
    for pair, (ia, ib) in pair_iso.items():
        if ia != ib:
            by_country[frozenset({ia, ib})].append(pair)
    derived = {cty for cty, crossings in by_country.items()
               if all(c in water for c in crossings)}
    if derived != {key for key, _br in adm0_roll}:
        fails.append("adm0 roll-up rows do not match the derivation from ADM1")

    return fails


def _load_engine():
    """The Stage 3-4 engine as a module (its input-file names are the contract)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("apply_overlays", HERE / "apply_overlays.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--full", action="store_true",
                    help="run the geometry build first (requires the GeoPackages)")
    ap.add_argument("--adm1-gpkg")
    ap.add_argument("--adm0-gpkg")
    ap.add_argument("--ocean-gpkg")
    ap.add_argument("--ndlsa-gpkg")
    ap.add_argument("--out-dir", help="output dir for --full (default: --data-dir)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir) if args.out_dir else args.data_dir

    if args.full:
        inputs = {"adm1_gpkg": args.adm1_gpkg, "adm0_gpkg": args.adm0_gpkg,
                  "ocean_gpkg": args.ocean_gpkg, "ndlsa_gpkg": args.ndlsa_gpkg}
        missing = [k for k, v in inputs.items() if not v]
        if missing:
            ap.error(f"--full requires {', '.join('--' + m.replace('_', '-') for m in missing)}")
        print("verifying input checksums against pins...")
        for key, path in inputs.items():
            digest = _sha256(Path(path))
            pin = PINNED_SHA256[key]
            if pin is None:
                print(f"  {key}: {digest}  (no pin recorded yet)")
            elif digest != pin:
                print(f"FATAL {key}: sha256 {digest} != pinned {pin}\n"
                      f"  A changed input is a data change: review it and update "
                      f"the pins here and in data/PROVENANCE.md.")
                return 1
            else:
                print(f"  {key}: OK")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-X", "utf8", str(HERE / "build_pericoupling_db.py"),
               "--adm1-gpkg", args.adm1_gpkg, "--adm0-gpkg", args.adm0_gpkg,
               "--ocean-gpkg", args.ocean_gpkg, "--ndlsa-gpkg", args.ndlsa_gpkg, "--out-dir", str(out_dir),
               "--clip-ocean"]
        print("running the geometry build, Stages 1-2 (this is the slow step)...", flush=True)
        subprocess.run(cmd, check=True)
        # the reviewed inputs of Stages 3-4 are INPUTS, not build outputs:
        # stage them into a scratch out-dir so the engine can apply them there
        overlays = _load_engine()
        if out_dir.resolve() != DEFAULT_DATA.resolve():
            import shutil
            for name in overlays.INPUT_FILES:
                shutil.copy(DEFAULT_DATA / name, out_dir / name)
        # a denylist entry whose contact the geometry no longer produces would
        # silently remove nothing: every entry must be present before Stage 4
        with open(out_dir / "pericoupled_adm1_edge_list.csv", newline="", encoding="utf-8-sig") as fh:
            built = {frozenset({r["ADM1_code_A"].strip(), r["ADM1_code_B"].strip()})
                     for r in csv.DictReader(fh)}
        with open(out_dir / overlays.DENYLIST_FILE, newline="", encoding="utf-8-sig") as fh:
            stale = [f"{r['code_a']}<->{r['code_b']}" for r in csv.DictReader(fh)
                     if frozenset({r["code_a"].strip(), r["code_b"].strip()}) not in built]
        if stale:
            print(f"FATAL denylisted contact(s) absent from the geometry build: {stale}")
            return 1
        print(f"  geometry build: {len(built)} ADM1 edges; every denylisted contact present")

    print("applying Stages 3-4 (the reviewed inputs)...", flush=True)
    engine = subprocess.run(
        [sys.executable, "-X", "utf8", str(HERE / "apply_overlays.py"),
         "--data-dir", str(out_dir)])
    if engine.returncode != 0:
        return engine.returncode

    print("verifying headline counts...")
    fails = verify_counts(out_dir)
    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    for k, v in EXPECTED.items():
        print(f"  {k}: {v} OK")

    if out_dir.resolve() == DEFAULT_DATA.resolve():
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "src/metacouplingllm/data"],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        print("byte-identity vs committed data: "
              + ("CLEAN (no differences)" if not status else f"DIFFERS:\n{status}"))
    print("build_all: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
