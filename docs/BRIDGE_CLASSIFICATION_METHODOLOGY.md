# Bridge / water-crossing classification — methodology, code & references

> **Note:** the OSM bridge-detection method below is the frozen discovery record for the base classification; current counts live in `docs/METHODS_adjacency.md` and `data/PROVENANCE.md`.

**Status:** verification complete & human-validated · **Date:** 2026-06-03/04
**Original run (historical base):** `bridge_classified_authoritative.csv` — **315 water-only ADM1 pairs: 108 with an open fixed crossing, 207 without.** *(The current base `bridge_classified_authoritative.csv` carries **301 rows** — 238 river / 63 lake, 101 with an open fixed crossing / 200 without — every river row re-adjudicated under the ru1 cross-vendor process (2026-07-21); the correction history, including the Italy↔Vatican land reclassification, is in `CHANGELOG.md`. Subsequent reviewed overlays extend the shipped `water_separated_pairs.csv` to **739 ADM1 pairs (345 with an open fixed crossing / 394 without)** plus **26 ADM0 roll-ups (20 bridge / 6 no-bridge)**: hydro-water 18, hydro-lakes 12, lake-gap 1, rescreen-water 386 (incl. the ru1-folded former wide-river and audit-water rows), rescreen-gap 21 (incl. the rg1-folded former river-gap rows, 2026-07-22). There is no lake *filter* — lakes are native exact-contact edges, so `coupling_standard` governs them natively like rivers, corrected only through the reviewed hydro-lakes/lake-gap overlays; see `data/PROVENANCE.md`. The 315 figures below describe the original OSM classification run.)*

This document records, in detail, how the bridge-classification database was built: why it
exists, the inputs, the OpenStreetMap (OSM) detection method and its evolution, the
independent multi-agent verification, the geometric confirmation, how everything was
reconciled, the resulting error taxonomy, the full file inventory, code, and references.

---

## 1. Purpose

The pericoupling DB marks two regions as adjacent ("pericoupled") when they share a border.
Some neighbours touch **only across a river or lake** — no land border. The project adds a
**`coupling_standard`** with three settings:

| standard | a water-only pair is pericoupled if… |
|---|---|
| **lenient** | always (any shared water counts) |
| **moderate** (default) | only if a **fixed land crossing** (bridge/causeway/dam-road/tunnel) links them |
| **stringent** | never (water never counts) |

So under the default **moderate** standard, the deciding fact for each water-only pair is:
**does a real, open, correctly-located fixed crossing exist between the two units?**
Motivating case: **Kinshasa ↔ Brazzaville** — ~9 km apart across the Congo, but no bridge
(ferry only), so they must *not* be pericoupled under moderate.

`has_bridge` is the column that drives this.

### Definition of a "fixed crossing" (the classification rule)
A pair is `has_bridge = True` iff there exists a **road or rail bridge, causeway, dam-top
road, or tunnel** that is

1. a real structure (not a riverbank road that merely runs near both banks),
2. **a structurally complete fixed link** — under-construction/proposed do NOT count; but a
   *completed* bridge on a politically-closed border DOES count (pericoupling is structural — see
   §12 border-closed policy),
3. **correctly located** — it actually connects *these two* ADM1 units, not a neighbouring one.

Ferries/boats are excluded (they are OSM relations, and are not fixed links).

---

## 2. Inputs

| input | source / version | role |
|---|---|---|
| **Water-only ADM1 candidate pairs** | derived in the pericoupling build (water-only labels are descriptive over native exact-contact edges — a ~2.5 km `ne_10m_rivers` centerline screen classifies river borders and lake borders derive from HydroLAKES; no lake filter exists). **OpenStreetMap is not used to detect water-only borders** (a prototype OSM water-polygon test was abandoned because rural riverbanks are unmapped); OSM supplies only the bridge flag below | the 315 pairs whose shared border is water with no land segment |
| **ADM1 unit polygons** | World Bank Official Boundaries ADM1 (2026-05-14), code field `ADM1CD_c` | the "both units" geometry for the crossing test |
| **`colab_candidates.gpkg`** (12.5 MB) | built from the above; layers `pairs` (315) + `units` (387) | the OSM-pipeline input |
| **OpenStreetMap** via **Overpass API** | live snapshot, 2026-06-03/04 | bridge/road/rail geometry |
| **Nominatim** | OSM geocoder | place→coordinate for the geometric confirmation |

Of the 315 candidates, **257 need OSM** (rivers + small/medium lakes) and **58 are huge
lakes** (Victoria, Tanganyika, Caspian-class, Great Lakes, etc.) auto-classified
`has_bridge = False` by knowledge (no bridge spans them) — flagged `needs_osm = False` in the gpkg.

---

## 3. OSM detection method (the "geometric" signal)

### 3.1 Core idea
For each pair, query OSM for bridge structures along/near the shared border, then count a way
as a crossing iff it **lies in both units** (offset-tolerant). Concretely, each unit polygon is
buffered by `UNIT_BUF = 0.0012°` (~130 m, the cross-source registration tolerance between OSM
ways and WB polygons), and a way is a crossing iff it intersects **both** buffers.

### 3.2 Why it evolved (v1 → v6) — the bugs and fixes
The public Overpass mirrors and OSM tagging forced six iterations (`colab_osm_classify_v*.py`):

| ver | change | problem it solved / introduced |
|---|---|---|
| v1 | per-pair OSM **water polygons** + roads | slow; rural riverbanks unmapped → wrong "land" reads |
| v2 | **bridge-only**, 3 concurrent workers rotating mirrors | concurrency hit the *same* mirror → rate-limit timeouts (136/315 `?`) |
| v3 | **serial**, bbox tiling | long borders still timed out (one huge bbox query) |
| v4 | **partial-success** tiles + drop heavy road-sweep on long borders | a flaky tile no longer abandons the whole pair |
| v5 | **3 mirrors, 1 worker each (fixed shards)** | ~3× faster, but a dead mirror stalled its whole shard |
| **v6** | **shared work-queue** (1 worker/mirror, healthy mirror drains tail) + **two false-positive fixes** | final method |

### 3.3 The two false-positive fixes in v6 (critical)
1. **Unnamed-road mode.** The "around" sweep along the border originally fetched *every*
   `way[highway]`/`way[railway]` near the border (no `[bridge]` filter), so an ordinary
   riverbank lane touching both 130 m buffers was counted as a crossing → false positives with
   `crossings="(unnamed)"`. **Fix:** the around-sweep is now bridge-filtered, and a post-filter
   `is_open_bridge()` requires a real bridge tag.
2. **Construction mode.** `way[highway][bridge]` also matches `highway=construction`, so
   not-yet-open bridges counted. **Fix:** `is_open_bridge()` drops `construction`/`proposed`.

### 3.4 Final method — key parameters
```
UNIT_BUF     = 0.0012   # ~130 m OSM↔WB registration tolerance (way must touch both unit buffers)
TILE_DEG     = 1.5      # split a long border's bbox into ≤1.5° tiles (keeps each query small)
TILE_IF_AREA = 2.0      # deg² above which we tile
SWEEP_IF_AREA= 1.0      # deg² below which we also run a bridge-filtered around-sweep (dikes/causeways)
AROUND       = 400      # m radius for the around-sweep
MAXPTS       = 45       # max sample points along the border for the around-sweep
TIMEOUT      = 35 s, 2 tries; 3 mirrors: overpass-api.de, kumi.systems, openstreetmap.fr
```

### 3.5 Final method — code (`colab_osm_classify_v6.py`)
```python
_NOT_OPEN = ("construction", "proposed")
def is_open_bridge(t):                         # count only real, OPEN bridge structures
    if t.get("highway") in _NOT_OPEN or t.get("railway") in _NOT_OPEN: return False
    if t.get("man_made") == "bridge": return True
    b = t.get("bridge")
    return bool(b) and b not in ("no", "false", "0")

def classify(row, home):                       # home = this worker's Overpass mirror index
    bd = row.geometry; ga, gb = ucode[row["code_a"]], ucode[row["code_b"]]
    b = bd.bounds; area = (b[2]-b[0]) * (b[3]-b[1])
    tl = tiles(b) if area > TILE_IF_AREA else [(b[1],b[0],b[3],b[2])]
    found = []; ok = 0; fails = 0
    for (s, w, n, e) in tl:                     # bridge structures over bbox tiles (no sampling gaps)
        bbox = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
        j = op(f'[out:json][timeout:{TIMEOUT}];'
               f'(way[highway][bridge]({bbox});way[railway][bridge]({bbox});'
               f'way["man_made"="bridge"]({bbox}););out geom tags;', home)
        if j is None: fails += 1; continue      # PARTIAL-SUCCESS: skip a flaky tile, keep the rest
        found += ways(j); ok += 1
    if area < SWEEP_IF_AREA:                     # compact borders: also bridge-filtered around-sweep (dikes)
        line = max(line_parts(bd), key=lambda x: x.length)
        pts = sample(line, MAXPTS)
        j2 = op(f'[out:json][timeout:{TIMEOUT}];'
                f'(way[highway][bridge](around:{AROUND},{pts});way[railway][bridge](around:{AROUND},{pts});'
                f'way["man_made"="bridge"](around:{AROUND},{pts}););out geom tags;', home)
        if j2 is not None: found += ways(j2); ok += 1
        else: fails += 1
    if ok == 0: return None, "", "osm-timeout"  # nothing came back at all → '?'
    gab, gbb = ga.buffer(UNIT_BUF), gb.buffer(UNIT_BUF)
    nm = sorted({tagname(t) for w, t in found
                 if is_open_bridge(t) and w.intersects(gab) and w.intersects(gbb)})  # ← crossing test
    note = "osm" if fails == 0 else f"osm-partial({fails}f/{len(tl)}t)"
    return (len(nm) > 0), "; ".join(nm[:5]), note
```
Concurrency: a `queue.Queue` of all pairs; 3 threads, thread `k` queries **only** mirror `k`
(≤1 concurrent query per mirror → no rate-limit). Resumable (checkpoints every row; `?` rows
retried on re-run). Output columns:
`code_a,name_a,country_a,code_b,name_b,country_b,water_type,water_body,border_km,ne_cov,has_bridge,crossings,note`.

### 3.6 OSM result lineage
- **Raw OSM (pre-fix):** 173 True / 142 False  (`bridge_classified_final.pre_aroundfix.bak`)
- **OSM-refix (v6 false-positive fixes, re-ran the 173 True rows):** 135 True / 180 False — **38 `True→False`** flips (the `(unnamed)`-road + construction artifacts). `bridge_classified_final.csv`.

The OSM signal alone is **geometry-only**: it cannot tell whether a real bridge sits in the
*wrong* province, whether it is *open*, or whether it was *missed* (remove-only re-query).
That is why an independent check was run.

---

## 4. Independent verification (the "real-world" signal)

A multi-agent workflow cross-checked **all 315** against pre-training knowledge + **web search**,
fully independent of OSM tagging.

### 4.1 Pass 1 — verify all 315  (`verify-bridge-pairs`, 96 agents, ~5.0 M tokens)
- 40 batch-agents (≈8 pairs each) judge each pair **independently** ("is there a fixed
  crossing between *province A* and *province B* across *water_body*?"), knowledge first then web.
- Every **disagreement with OSM** gets a focused **adversarial recheck** (a second agent that
  web-verifies the ground truth and cites sources).
- **Result:** 259/315 agree with OSM; 56 flagged; **51 confirmed errors** (48 false-positives
  `True→False`, 3 false-negatives `False→True`); 5 rechecks upheld OSM; 0 left unsure.

Structured output schema (forces a committed verdict, validated at the tool layer):
```jsonc
// per pair
{ "pair":"code_a|code_b", "our_has_bridge":bool,
  "independent_verdict":"crossing_exists|no_crossing|unsure",
  "agrees":bool, "confidence":"high|medium|low", "evidence":"…named crossing or basis…" }
```

### 4.2 Pass 2 — resolve the 14 conflicts  (`verify-bridge-conflicts`, 14 agents)
The OSM-refix (geometric) and Pass-1 web disagreed on 14 pairs (OSM dropped them as
`(unnamed)` road, but the first-pass web said a crossing exists). These were **not** caught by
Pass 1's recheck because they agreed with the *old* OSM. A dedicated agent per pair resolved
each, checking **two things explicitly**:
```jsonc
{ "pair":"…", "has_open_crossing":bool,
  "open_to_traffic":bool,      // reject construction/proposed
  "correct_province":bool,     // bridge must land in BOTH named units, not a neighbour
  "confidence":"…", "evidence":"…with sources…" }
```
**Result:** 5 → True (real bridges OSM had mistagged as plain road/rail), 9 → False
(real bridge but wrong province / not open / proposed).

---

## 5. Geometric confirmation of the 14 conflicts (third, independent signal)

To validate the conflict calls without any LLM reasoning: geocode the named crossing
(**Nominatim**) → test the point against the **province polygons** (nearest-province +
distance to each candidate unit). This answers deterministically: *which two provinces does
this crossing physically connect?*
```python
from shapely.geometry import Point; from shapely.ops import nearest_points; from pyproj import Geod
geod = Geod(ellps="WGS84")
def dist_km(pt, poly):
    a, b = nearest_points(pt, poly); return geod.inv(a.x, a.y, b.x, b.y)[2] / 1000.0
# geocode crossing → pt; rank all unit polygons by distance to pt;
# connects_this_pair := dist_km(pt, code_a) < ~3 and dist_km(pt, code_b) < ~3
```
**Result:** 12/14 confirmed; 2 flagged (`LTU007↔RUS023`, `ZAF005↔BWA005`) → **manually
reviewed by the maintainer and the conclusion/evidence confirmed correct**. It proved the
subtle wrong-province calls, e.g.:
- Stichtse Brug → Flevoland↔**Noord-Holland (NLD008)**, not Utrecht (NLD010)
- Jikaw Bridge → Gambela↔**Upper Nile (SSD009)**, not Jonglei (SSD003)
- Calueque Dam → Cunene↔**Omusati (NAM010)**, not Kunene (NAM007)
- Poldasht–Shah Takhti → W.Azarbayejan↔**Kengerli (AZE035)**, not Sadarak (AZE054)
- Ruzizi bridge → Sud-Kivu↔**Rwanda (RWA005)**, not Cibitoke (BDI006)
- `ZWE006↔ZMB109` True confirmed — it is the **Kariba Dam** (not Chirundu, a neighbour pair)

---

## 6. Reconciliation → authoritative dataset

The authoritative dataset starts from the OSM-refixed base and layers the verified corrections:
```
authoritative = OSM-refix (135/180)
   + web false-positives  (24 confirmed) → False     # incl. real-bridge-tag misattributions OSM kept
   + web false-negatives  (3)            → True       # real bridges OSM missed
   + conflict verdicts    (5 True, 9 False)
```
**Lineage:** raw OSM **173/142** → OSM-refix **135/180** → reconciled **119/196** → after the
extra verification sweep (§12) **108/207**.
Net: spurious "bridges" removed, real ones recovered, province/open-status corrected.

Provenance columns carried in the audit: `raw_osm → osm_refix → authoritative`, plus
`decided_by`, `agreement` (`both` 24 / `web-only` 27 / `conflict-resolved` 14 / `osm+web agree` 250),
`reason`, `confidence`, `evidence`.

---

## 7. Error taxonomy (why OSM was wrong ~16% of the time)

| mode | meaning | example |
|---|---|---|
| **province misattribution** (dominant) | a real bridge caught by the 130 m buffer that actually lands in the *neighbouring* ADM1 | 5th Mekong Friendship Bridge → Bueng Kan not Nong Khai; Stichtse Brug |
| **street-artifact** | a riverbank lane within both buffers, no real span | Amazon `BRA004↔PER016` "Beco Cinco de Setembro" (a town alley) |
| **ferry/boat-only** | crossing is by boat, no fixed link | Senegal R. (Gorgol↔Matam); Rio Coco |
| **not-open** | bridge real but under construction | Bioceanic Bridge (opens 2026-09); Rosso |
| **abandoned/unbuilt** | structure never finished | Akobo bridge (ETH↔SSD) |
| **internal mislabeled international** | a within-country bridge | Wiwilí bridge (Nicaragua-internal) on the Coco |
| **recovered (false-negative)** | real bridge OSM tags as plain road/rail/dam | Queen Louise; Kariba Dam; Botovo rail bridge; Yacyretá |

Note: the 130 m buffer biases toward **false-positives** (over-keep), which is the *safe*
direction for a connectivity dataset; the verification removed those.

---

## 8. File inventory (under `build_data/`; untracked by git except
`bridge_classified_authoritative.csv`, tracked since PR #89 as the pinned
`--full`-rebuild input for validation-study reproducibility)

**Inputs / pipeline**
- `colab_candidates.gpkg` — 315 pairs + 387 units (OSM-pipeline input)
- `colab_osm_classify_v6.py` — final classifier (shared-queue, 3 mirrors, `is_open_bridge`)
- `colab_osm_classify_v3.py … v5.py` — superseded iterations (share the pre-v6 around-sweep bug)
- `bridge_regen_final.py` — re-ran the 173 True rows with the v6 fix (remove-only, fails==0 guard)

**Data (lineage)**
- `bridge_classified_final.pre_aroundfix.bak` — raw OSM (173/142)
- `bridge_classified_final.csv` — OSM-refix (135/180)
- `bridge_classified_authoritative.csv` — **final merged: 119/196, then 108/207 after the §12 sweep**

**Audit / provenance**
- `bridge_corrections.csv` — the 51 web corrections (direction, confidence, evidence)
- `bridge_audit_full.csv` — 56 changed pairs (source, confidence, evidence)
- `bridge_audit_complete.csv` — **all 315** with `raw→refix→final`, `decided_by`, `agreement`, `reason`, `evidence`
- `bridge_regen_changelog.csv` — the 38 OSM-refix flips
- `BRIDGE_FIX_HANDOFF.md` — the v6 false-positive fix write-up
- this file — full methodology

**Verification workflows** (transcripts under `…/subagents/workflows/`)
- `verify-bridge-pairs` (run `wf_6ee09481…`) — 315-pair web verification
- `verify-bridge-conflicts` (run `wf_cf7fc695…`) — 14 conflict resolutions

---

## 9. References

**Data sources**
- OpenStreetMap contributors — way/relation geometry & tags. © OpenStreetMap, ODbL.
- Overpass API — `https://overpass-api.de`, `https://overpass.kumi.systems`, `https://overpass.openstreetmap.fr`.
- Nominatim — `https://nominatim.openstreetmap.org` (geocoding; usage-policy compliant: 1 req/s, UA set).
- Natural Earth — `ne_10m_rivers`, `ne_10m_lakes` (descriptive water-type classification only — a ~2.5 km `ne_10m_rivers` centerline screen labels river borders; `ne_10m_lakes` does not detect lake borders and no lake filter exists). OpenStreetMap is used only for the bridge flag, not to detect water-only borders.
- HydroLAKES — the lake data source for lake borders (`hydro_lakes_overlay_pairs.csv`, from a HydroLAKES full-database sweep).
- World Bank Official Boundaries — ADM1 (2026-05-14), field `ADM1CD_c`.

**Method notes**
- OSM tagging keys used: `bridge`, `man_made=bridge`, `highway`, `railway`, `highway=construction|proposed`.
- Crossing test = offset-tolerant "way intersects both unit buffers"; ferries (relations) excluded by construction.
- Per-pair real-world evidence (named bridges, opening dates, ferry/boat confirmations, official
  border-post docs) is cited inline in `bridge_corrections.csv` / `bridge_audit_complete.csv`
  `evidence` columns — e.g. Wikipedia, TYPSA (Wiwilí), WFP Logistics Cluster (Porga), Eye Radio
  (Akobo), RailwayPro/trans.info (Botovo–Gyékényes), OMVS/SOGENAV (Senegal R. ferries).

---

## 10. Limitations & provenance

1. **Point-in-time OSM snapshot** (2026-06-03/04). New bridges (e.g. Bioceanic, opening
   2026-09) will need re-checking; record the snapshot date with the shipped data.
2. **Not build-reproducible.** The dataset embeds web + manual verification, so the build
   script can document but **cannot regenerate** it from geometry alone. It must ship as a
   **reviewed static artifact** (like the disputed-territory overlay), with this provenance.
3. **Residual risk.** The 259 OSM↔web agreements were not adversarially re-checked (two
   independent methods already concurred — strong but not infallible). The 51 corrections + 14
   conflicts were triple-checked (web + adversarial recheck + geometry + maintainer review).
4. **"Open to traffic" rule** is applied as of the snapshot date — under-construction links are
   `False` until they open.
5. **130 m buffer** can false-positive at narrow town channels / tripoints; those were the cases
   the verification targeted.

---

## 11. Final state & how it feeds `coupling_standard`

> **Note (2026-07).** The verification pattern this document records — OSM
> screen → independent web verification → adversarial recheck of
> disagreements → geocode + province-polygon cross-check — was later
> formalized as the **four-layer bridge pipeline**
> (`docs/METHODS_adjacency.md` §8) and reused for every rescreen-water /
> rescreen-gap addition, which now supply the majority of the 739 shipped
> water-only pairs.

`bridge_classified_authoritative.csv` → frozen as a dated **`water_separated_pairs.csv`**
(`has_bridge` per pair). Applied to the 315 water-only ADM1 pairs of this run:

| standard | kept (pericoupled) | dropped |
|---|---|---|
| lenient | 315 | 0 |
| **moderate** (default) | **108** (have an open fixed crossing) | 207 |
| stringent | 0 | 315 |

ADM0 roll-up: a country pair is water-only only if *all* its province crossings
are water-only; it "has a bridge" if *any* does. Most country pairs have some land border and
are unaffected.

*(Current shipped state after the reviewed overlays: `water_separated_pairs.csv` carries
**739 ADM1 pairs (345 / 394)** plus **26 ADM0 roll-ups (20 bridge / 6 no-bridge)**;
the base `bridge_classified_authoritative.csv` is **301 rows**. See
`data/PROVENANCE.md`.)*

---

## 12. Extra verification sweep (2026-06-04) — final reconciliation to 108/207

After the 119/196 reconciliation, two further independent passes closed the residual gap (the 259
first-pass agreements had only been single-checked):

- **#1 — geometric province check on all 119 `True`** (`geocheck_true.py`): re-query OSM bridge
  geometry; confirm a matched bridge sits on THIS pair's shared border (≤1.5 km). 111/111
  OSM-tagged + 8/8 recovered confirmed on-border. *Caveat:* the 130 m buffer is too coarse for
  fine-grained units (Liechtenstein municipalities) and tri-points, so #1 alone over-confirms.
- **#2 — adversarial web recheck of the agreed set** (`recheck-agreed-set` workflow, 25 agents):
  the 111 agreed-`True` (province + open-to-traffic) and 82 agreed-`False` rivers (missed
  crossing). Flagged **18 `True→False` + 3 `False→True`** — adding the dimension neither prior
  pass tested: **is the border politically open**.
- **Geocode adjudication** of the province flags + the 3 reversals resolved them deterministically
  — and **overruled #2 on 3** (no single method is perfect): `ZMB105↔ZWE006` (WB polygons place
  Chirundu in Lusaka → keep True), `BGR018↔ROU020` (Friendship Bridge is Ruse not Silistra → keep
  False), `ARG007↔URY015` (Salto Grande is Entre Ríos not Corrientes → keep False).

**Applied (11 `True→False`):** 3 under-construction (Rosso, Aghband-Kalala, Black Volta), 1
no-bridge (Komadugu Yobe), 2 wrong-province (Sher Khan Bandar→Kunduz; Arizona→Sonora), 5 fine-unit
Rhine/Doubs (Triesen, Eschen, Ruggell↔Austria, Vaud-Rhône, Doubs-Bern). **Flagged for review
(kept):** 2 tri-points (`IND005↔NPL001`, `ZMB109↔ZWE011`). Changelog: `recheck2_changes.csv`;
pre-sweep backup `bridge_classified_authoritative.pre_recheck2.bak`.

### Border-closed policy (definitional decision, locked)
5 pairs have a **real, correctly-located, completed** bridge whose **border is politically closed**
(Armenia–Turkey since 1993, Tajik–Afghan 2026, Niger–Benin 2023, Narva vehicles 2024, Khudafarin).
**Decision: these count as pericoupled (`True`).** Rationale: pericoupling is a *structural*
relation — the physical link exists; closures are transient policy; and political openness was not
checked for any other pair (incl. all land borders), so excluding only these would be inconsistent.
"Open to traffic" is interpreted as **"the fixed link is structurally complete"** (excludes
under-construction), not "currently politically open".

**Residual-risk floor:** even after three independent signals (OSM geometry + 2 web passes +
geocode adjudication + maintainer review), the hardest ~1% — fine-grained units, tri-points,
boundary-vintage mismatches — needs human judgment. The shipped CSV is a reviewed static artifact
and those rows are flagged for easy correction.
