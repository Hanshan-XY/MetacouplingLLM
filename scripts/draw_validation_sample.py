# -*- coding: utf-8 -*-
"""Draw the preregistered validation sample (docs/VALIDATION_SAMPLING_PLAN.md).

Fully deterministic: fixed seed 20260713, strata sorted by pair id before
drawing. Re-running reproduces the identical sample byte-for-byte.

Inputs: the shipped rescreen manifests (population P) and the untracked
audit-evidence tree `build_data/water_screen_rebuild/` (populations N1/N2 —
frozen verdict JSONs and the mechanism ledger). Outputs (untracked, for the
rater): validation worksheets + a drawn-ids manifest under
build_data/water_screen_rebuild/validation/.

Run:  python scripts/draw_validation_sample.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "src" / "metacouplingllm" / "data"
W5 = REPO / "build_data" / "water_screen_rebuild"
OUT = W5 / "validation"
SEED = 20260713

# ---------------- population P: shipped dual-AI rescreen rows ----------------
P = []
for fname, batch_of in [
    ("rescreen_water_overlay_pairs.csv", None),
    ("rescreen_gap_overlay_pairs.csv", None),
]:
    for r in csv.DictReader(open(DATA / fname, newline="", encoding="utf-8-sig")):
        if "dual-AI" not in r["source"]:
            continue
        src = r["source"]
        batch = ("b1" if "rebuild:" in src or "rebuild b1" in src else
                 "b2" if " b2:" in src else
                 "b3-b6" if "b3-b6" in src else
                 "holds" if "20km-hold" in src else "b1")
        P.append({
            "pair": f"{r['code_a']}<->{r['code_b']}",
            "name_a": r["name_a"], "iso_a": r["iso_a"],
            "name_b": r["name_b"], "iso_b": r["iso_b"],
            "population": "cross-border" if r["iso_a"] != r["iso_b"] else "domestic",
            "water_type": r["water_type"], "water_body": r["water_body"],
            "batch": batch,
        })
print(f"P (dual-AI shipped rows): {len(P)}")

# ---------------- populations N1/N2: adjudicated & mechanism negatives -------
def fmax(m):
    vals = []
    for k in ("river_f", "lake_f", "river_f_hydrorivers_500m", "lake_f_hydrolakes_500m"):
        v = m.get(k)
        if v not in (None, ""):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return max(vals) if vals else 0.0

N1 = []
for f in ("b1", "b2", "b3b4", "b5b6", "holds20km"):
    for r in json.load(open(W5 / "verdicts" / f"{f}.json", encoding="utf-8"))["results"]:
        if r["research"]["water_only"] or r["judgment"]["water_only"]:
            continue                      # both-false only (never human-reviewed)
        m = r["meta"]
        N1.append({
            "pair": r["pair"],
            "name_a": m["name_a"], "iso_a": m["iso_a"],
            "name_b": m["name_b"], "iso_b": m["iso_b"],
            "population": "cross-border" if m.get("cross") in ("True", True) else "domestic",
            "water_type": r["judgment"].get("water_type", ""),
            "water_body": str(r["judgment"].get("water_body", ""))[:80],
            "fmax": round(fmax(m), 2),
            "batch": f,
        })
n1_high = [x for x in N1 if x["fmax"] >= 0.50]
n1_rest = [x for x in N1 if x["fmax"] < 0.50]
print(f"N1 (both-false): {len(N1)} = high {len(n1_high)} + rest {len(n1_rest)}")

N2 = []
for r in csv.DictReader(open(W5 / "hybrid_autoreject_ledger.csv",
                             newline="", encoding="utf-8-sig")):
    N2.append({
        "pair": f"{r['a']}<->{r['b']}",
        "name_a": r["name_a"], "iso_a": r["iso_a"],
        "name_b": r["name_b"], "iso_b": r["iso_b"],
        "population": "cross-border" if r.get("cross") in ("True", True) else "domestic",
        "water_type": "", "water_body": "", "fmax": "",
        "batch": "mechanism",
    })
print(f"N2 (mechanism ledger): {len(N2)}")

# ---------------- draws ------------------------------------------------------
rng = random.Random(SEED)

def draw(pool, n):
    pool = sorted(pool, key=lambda x: x["pair"])
    return pool if len(pool) <= n else rng.sample(pool, n)

# P: proportional allocation by (batch, population, water_type), largest remainder
strata: dict[tuple, list] = {}
for x in P:
    strata.setdefault((x["batch"], x["population"], x["water_type"]), []).append(x)
keys = sorted(strata)
quota = {k: 80 * len(strata[k]) / len(P) for k in keys}
alloc = {k: max(1, int(quota[k])) for k in keys}
while sum(alloc.values()) < 80:
    k = max(keys, key=lambda k: (quota[k] - alloc[k], k))
    alloc[k] += 1
while sum(alloc.values()) > 80:
    k = min((k for k in keys if alloc[k] > 1), key=lambda k: (quota[k] - alloc[k], k))
    alloc[k] -= 1
p_sample = []
for k in keys:
    p_sample += draw(strata[k], alloc[k])
n_sample = draw(n1_high, 40) + draw(n1_rest, 40) + draw(N2, 20)
for x, s in [(p_sample, "P"), (n_sample, None)]:
    pass
for x in p_sample:
    x["stratum"] = f"P:{x['batch']}:{x['population']}:{x['water_type']}"
for x in n_sample:
    x["stratum"] = ("N2:mechanism" if x["batch"] == "mechanism" else
                    f"N1-{'high' if float(x['fmax']) >= 0.5 else 'rest'}:{x['batch']}")

# inter-rater subsample: 15 + 15 from the drawn sets
irr = set(x["pair"] for x in rng.sample(sorted(p_sample, key=lambda x: x["pair"]), 15))
irr |= set(x["pair"] for x in rng.sample(sorted(n_sample, key=lambda x: x["pair"]), 15))

# ---------------- worksheets --------------------------------------------------
OUT.mkdir(exist_ok=True)
COLS = ["stratum", "pair", "name_a", "iso_a", "name_b", "iso_b", "population",
        "screen_water_type", "screen_water_body", "fmax", "second_rater",
        "genuine_water_only", "notes"]

def write(rows, fname):
    with open(OUT / fname, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for x in sorted(rows, key=lambda x: (x["stratum"], x["pair"])):
            w.writerow({
                "stratum": x["stratum"], "pair": x["pair"],
                "name_a": x["name_a"], "iso_a": x["iso_a"],
                "name_b": x["name_b"], "iso_b": x["iso_b"],
                "population": x["population"],
                "screen_water_type": x["water_type"],
                "screen_water_body": x["water_body"],
                "fmax": x.get("fmax", ""),
                "second_rater": "yes" if x["pair"] in irr else "",
                "genuine_water_only": "", "notes": "",
            })
    print(f"wrote {fname}: {len(rows)} rows")

write(p_sample, "worksheet_validation_P.csv")
write(n_sample, "worksheet_validation_N.csv")
with open(OUT / "drawn_ids.json", "w", encoding="utf-8", newline="\n") as fh:
    json.dump({"seed": SEED,
               "P": sorted(x["pair"] for x in p_sample),
               "N": sorted(x["pair"] for x in n_sample),
               "second_rater_subsample": sorted(irr)}, fh, indent=1)
print(f"P sample {len(p_sample)} | N sample {len(n_sample)} | second-rater {len(irr)}")
