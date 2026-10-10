#!/usr/bin/env python
"""Build the EEZ -> world-region map used for the regional tables (the paper's Tables 4 and 5).

The 280 EEZs (the focal systems of the edge table) carry
finer-grained names than the 168 nations in Carlson et al. (2020) Table S16
(e.g. "USA (Alaska, Subarctic)", "Russia (Far East)", "Korea (South)",
"Hawaii Main Islands (USA)").  This step joins each EEZ to its nation and then
to that nation's "World region" exactly as Carlson assigned it in Table S16, so
the regional tables follow Carlson et al.'s own regional scheme.

Inputs (both pinned under data/, fully offline):
    data/s16_country_region.json   -- {nation: "World region"} from Carlson Table S16 (168 nations)
    data/fishing_edge_table.csv    -- the 280 focal EEZ names (built by 00_build_edge_table.py)

Output:
    data/eez_region.json           -- {EEZ: {grp, region}}, grp = readable label, region = raw S16

Join logic (nation-political, matching S16's nation-level aggregation):
  1. direct name / alias match on the EEZ name and its pre-parenthetical base;
  2. for sub-national EEZs, the parenthetical parent nation ("... (USA)", "... (Spain)");
  3. Korea (South)/(North) disambiguated explicitly.
Genuinely non-S16 entities (Taiwan -- excluded by Carlson for lack of a World Bank
GDP value -- plus three minor territories: Cape Verde, Cook Islands, Ascension)
remain unmapped; they carry no region in the paper's scheme and are dropped from
the regional means, exactly as in Carlson's Table S16.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REG = json.loads((DATA / "s16_country_region.json").read_text(encoding="utf-8"))

# Readable label for each raw S16 "World region".
LABEL = {
    "Africa_western": "Africa (West)",
    "Africa_not western": "Africa (non-western)",
    "Asia": "Asia",
    "Caribbean": "Caribbean",
    "Central America": "Central America",
    "Europe_eastern/southern": "Europe (E/S)",
    "Europe_northern": "Europe (North)",
    "Europe_western": "Europe (West)",
    "North America": "North America",
    "Oceania": "Oceania",
    "South America": "South America",
}

STOP = {"and", "the", "of", "isl", "islands", "island", "rep", "st", "saint", "main"}


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    return "".join(t for t in re.findall(r"[a-z]+", s) if t not in STOP)


S16_NORM = {norm(k): k for k in REG}

# EEZ-style name (normalised) -> S16 canonical nation.
ALIAS = {
    "usa": "United States", "russia": "Russian Federation", "vietnam": "Vietnam",
    "iran": "Iran, Islamic Rep.", "egypt": "Egypt, Arab Rep.", "syria": "Syrian Arab Republic",
    "yemen": "Yemen, Rep.", "brunei": "Brunei Darussalam", "gambia": "Gambia, The",
    "bahamas": "Bahamas, The", "ivorycoast": "Cote d'Ivoire", "cotedivoire": "Cote d'Ivoire",
    "micronesia": "Micronesia, Fed. Sts.", "congoexzaire": "Congo, Dem. Rep.",
    "congordc": "Congo, Dem. Rep.", "congorof": "Congo, Rep.", "hongkong": "Hong Kong SAR, China",
    "timorleste": "Timor-Leste", "easttimor": "Timor-Leste", "faeroe": "Faroe Islands",
    "uk": "United Kingdom", "frenchguiana": "France", "reunion": "France", "mayotte": "France",
    "usvirgin": "Virgin Islands (U.S.)", "gazastrip": "West Bank and Gaza",
}


def s16_lookup(name: str) -> str | None:
    n = norm(name)
    if n in ALIAS:
        return ALIAS[n]
    return S16_NORM.get(n)


def nation(eez: str) -> str | None:
    low = eez.lower()
    if low.startswith("korea"):
        return "Korea, North" if "north" in low else "Korea, South"
    base = re.split(r"[(,]", eez)[0].strip()
    for cand in (eez, base):
        nat = s16_lookup(cand)
        if nat:
            return nat
    for paren in re.findall(r"\(([^)]+)\)", eez):   # try every parenthetical parent
        nat = s16_lookup(paren.strip())
        if nat:
            return nat
    return None


def main() -> None:
    eezs = sorted({r["focal_system_id"]
                   for r in csv.DictReader((DATA / "fishing_edge_table.csv").open(encoding="utf-8"))})
    out, mapped = {}, 0
    for eez in eezs:
        nat = nation(eez)
        if nat and nat in REG:
            raw = REG[nat]
            out[eez] = {"grp": LABEL[raw], "region": raw}
            mapped += 1
        else:
            out[eez] = {"grp": None, "region": None}
    (DATA / "eez_region.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote data/eez_region.json: {mapped}/{len(eezs)} EEZs mapped to a region")
    unmapped = [e for e in eezs if out[e]["grp"] is None]
    print(f"unmapped ({len(unmapped)}, all non-S16): {', '.join(unmapped)}")


if __name__ == "__main__":
    main()
