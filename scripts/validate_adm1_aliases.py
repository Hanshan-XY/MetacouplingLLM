#!/usr/bin/env python
"""Validate and write the ADM1 alias CSVs from raw workflow output.

Reads build_data/raw_aliases.json (output of the alias generation workflow),
adds the curated aliases listed in this script (_CURATED_ALIASES, source
"curated"), applies all deterministic validation rules, and writes:

  src/metacouplingllm/data/adm1_aliases.csv   — lean shipped table
  docs/adm1_aliases_audit.csv                 — full provenance table

Validation rules
----------------
N0  Normalize key: lowercase, strip, collapse spaces, NFKD-fold.
G   Drop if the key IS a country's ISO code, name or alias
    (resolve_country_code with substring=False), the names resolve_adm1_code
    refuses without a country hint.  A key that merely contains a country
    name ("upper austria", "mexico city") is kept and resolves with or
    without a hint.
S1  Drop keys shorter than 4 characters.
W1  Drop generic single geo-words (direction/type words, ADM1 suffixes).
DR  Drop curated manual-review rejections (_REVIEW_DENYLIST): semantic scope
    violations (historical/former names, capital-city namesakes, multi-region
    labels, acronyms) the deterministic rules cannot detect.  From the Opus
    double-check of PR #60 (see PR #61), and from the review of the aliases
    rule G admitted when it stopped dropping keys that contain a country
    name (2026-10-16).
R1  Drop if resolve_adm1_code(key, country=iso) already returns the same code
    (redundant — the existing resolver already handles it), unless the key
    is one word that the resolver found without a hint only as part of a
    longer name ("beijing" in "Beijing Shi"), which its single-word guard
    refuses since 2026-10-16.
O1  If the resolver (with hint) returns a DIFFERENT non-None code → mark as
    override; exclude from the shipped CSV, list in the audit for review.
X1  Drop if resolve_adm1_code(key) with NO hint returns a different non-None
    code (fold-aware no-hint shadow — Strategy 0 runs first and would otherwise
    shadow another region's natural resolution; catches folded cross-country
    collisions U3 misses).
U1  Intra-country: if two aliases in the same country share a key → drop both.
U2  Cross-country: if a key maps to regions in more than one country → drop
    (global key ambiguity; callers would need a country hint, reducing value).
U3  Drop if key matches an existing canonical index entry for a different code
    (would shadow a real region name).
D0  Deduplicate identical (code, alias_key) pairs from parallel generation;
    a curated record wins over a generated one.
C1  Cap at 3 generated aliases per code (highest confidence first, then
    alphabetical); curated aliases always ship.

R1, O1 and X1 test a key against the name matcher that the alias table
supplements: _resolve_adm1_name with the alias table cleared (strategies 1-3
of resolve_adm1_code), not resolve_adm1_code itself, whose trailing-designator
retry and spelling-key comparison run after the alias table.  Aliases such as
"rajshahi division" and "taita-taveta" therefore stay in the table although
the retry or the spelling keys would also resolve them.

The script stops with an error if any rule drops a curated alias.

Usage
-----
    python scripts/validate_adm1_aliases.py [--raw build_data/raw_aliases.json]

Run after the alias-generation workflow completes and
build_data/raw_aliases.json has been written.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metacouplingllm.knowledge.adm1_pericoupling import (  # noqa: E402
    _load_adm1_data,
    _get_adm1_name_index,
    _resolve_adm1_name,
)
from metacouplingllm.knowledge.countries import resolve_country_code  # noqa: E402

# ---------------------------------------------------------------------------
# Generic geo-words to drop (W1)
# ---------------------------------------------------------------------------
_W1_WORDS: frozenset[str] = frozenset({
    "north", "south", "east", "west", "central", "upper", "lower",
    "new", "old", "great", "greater", "inner", "outer",
    "region", "province", "state", "city", "county", "district",
    "department", "territory", "island", "islands", "coast",
    "sheng", "oblast", "okrug", "kray", "pradesh", "laen",
    "novads", "maakond", "zizhiqu", "rep", "giang",
})

# ---------------------------------------------------------------------------
# DR: manual review denylist (Opus double-check of PR #60, see PR #61)
# ---------------------------------------------------------------------------
# The deterministic rules above cannot detect *semantic* scope violations that
# leaked from the LLM generation prompt: historical/former names, capital-city-
# for-region namesakes, multi-region cultural labels, and acronyms that slipped
# the S1 length floor.  A full Opus audit of all 1,178 generated aliases (each
# finding independently verified) confirmed the (alias_key, code) pairs below as
# out-of-scope or wrong-region.  Four still-extremely-common former names were
# deliberately KEPT (saigon→VNM029, rangoon→MMR015, orissa→IND026,
# pondicherry→IND027) as a curated allowance, so they are absent here.
# Keys are NFKD-folded lowercase, matching ``_normalize_key``.
_REVIEW_DENYLIST: frozenset[tuple[str, str]] = frozenset({
    # wrong_region (points at a different/adjacent place than the WB region)
    ("jubaland", "SOM010"),
    ("loire valley", "FRA019"),
    ("ust-orda buryat autonomous okrug", "RUS002"),
    ("auvergne-rhone-alpes", "FRA023"),
    ("cappadocia", "TUR060"),
    ("windhoek region", "NAM006"),
    # namesake / capital-city-for-region / sub-region
    ("balkanabat region", "TKM002"),
    ("golan", "SYR012"),
    ("kavango east", "NAM005"),
    ("kavango west", "NAM005"),
    ("mosul", "IRQ015"),
    ("nasiriyah", "IRQ017"),
    ("new delhi", "IND010"),
    ("okavango", "BWA007"),
    ("santo domingo de los colorados", "ECU023"),
    ("south tyrol", "ITA017"),
    # too_generic / multi-region label / feature-not-region
    ("plain of jars", "LAO018"),
    ("basseterre parish", "KNA003"),
    ("central province", "IND020"),
    ("westfjords", "ISL022"),
    # historical / former names (excluded by v1 scope)
    ("antioch", "TUR036"),
    ("caesarea", "TUR047"),
    ("dacca", "BGD003"),
    ("diego suarez", "MDG030"),
    ("dvinsk", "LVA025"),
    ("iconium", "TUR053"),
    ("kirovabad", "AZE019"),
    ("kronstadt", "ROU009"),
    ("mangyshlak region", "KAZ010"),
    ("mocamedes", "AGO016"),
    ("mysore", "IND017"),
    ("north atlantic autonomous region", "NIC014"),
    ("nukha", "AZE059"),
    ("san blas", "PAN008"),
    ("smyrna", "TUR041"),
    ("south atlantic autonomous region", "NIC015"),
    ("tamim", "IRQ013"),
    ("tenasserim", "MMR014"),
    ("trebizond", "TUR074"),
    ("united provinces", "IND034"),
    ("uttaranchal", "IND035"),
    ("west sepik", "PNG018"),
    ("zabaykalsky krai", "RUS014"),
    # other (false rename claim / acronyms / nicknames / over-literal calques)
    ("kropyvnytskyi oblast", "UKR011"),
    ("gbao", "TJK001"),
    ("raccn", "NIC014"),
    ("raccs", "NIC015"),
    ("snnpr", "ETH010"),
    ("bicolandia", "PHL005"),
    ("mother of god", "PER017"),
    ("silent city", "MLT029"),
    # Review of 2026-10-16.  Rule G had dropped every key containing a
    # country name; when it came to drop only keys that ARE one, it admitted
    # 50 generated aliases.  Each was checked against Wikipedia (its lists of
    # the countries' divisions, the units' own articles); 38 shipped and
    # these 12 did not.
    # capital-city namesakes: the city is part of the unit
    ("kuwait city", "KWT004"),
    ("luxembourg city", "LUX003"),
    ("panama city", "PAN011"),
    # sub-region: the Riviera is the coast of Provence-Alpes-Cote d'Azur
    ("french riviera", "FRA022"),
    # generic or multi-region labels: "northern Kazakhstan" spans several
    # oblasts (the Virgin Lands); "southwestern Ethiopia" studies in the
    # RAG corpus lie in Jimma Zone, Oromia; "North Lebanon" holds the North
    # and Akkar governorates, Southern Lebanon the South and Nabatieh ones
    ("northern kazakhstan", "KAZ012"),
    ("eastern kazakhstan", "KAZ013"),
    ("western kazakhstan", "KAZ015"),
    ("southwest ethiopia", "ETH012"),
    ("north lebanon", "LBN007"),
    ("south lebanon", "LBN008"),
    # former name: Bas-Congo, renamed Kongo Central in 2015
    ("lower congo", "COD002"),
    # over-literal calque, and the name of colonial Mauritius (1715-1810)
    ("isle of france", "FRA013"),
    # Second review of 2026-10-16.  Since resolve_adm1_code stopped taking
    # one word without a country hint as part of a longer name, R1 admits
    # the 50 generated one-word aliases that had resolved only that way
    # ("beijing" for Beijing Shi, "michoacan" for Michoacán de Ocampo).
    # Three are common English words that a text scan would mostly misread:
    # "northwest" (72 times in the RAG corpus, 47 of them lowercase; it had
    # given Canada's Northwest Territories), "oriental" and "udon" (the
    # noodle).
    ("northwest", "HTI007"),
    ("oriental", "MAR006"),
    ("udon", "THA072"),
})

# ---------------------------------------------------------------------------
# Curated aliases, added on 2026-10-16 after the generation run
# ---------------------------------------------------------------------------
# English names, and local spellings the World Bank names abbreviate or
# misspell, that researchers write and the resolver did not match, or matched
# only as the start of a longer name ("Brussels").  Each name
# was checked against the source quoted in its basis.  "Mexico City", "Lesser
# Poland", "Greater Poland" and "Magallanes and the Chilean Antarctica"
# contain a country name or demonym but are not one, so rule G keeps them.
_CURATED_ALIASES: tuple[dict[str, str], ...] = (
    {"alias_display": "Mexico City", "code": "MEX009", "iso_a3": "MEX",
     "basis": "English name of Mexico's capital, which the WB stores under "
              "its Spanish name (Wikipedia, 'Mexico City': the federal "
              "district, renamed Ciudad de México in 2016)."},
    {"alias_display": "Silesian Voivodeship", "code": "POL012", "iso_a3": "POL",
     "basis": "English name of śląskie (Wikipedia, 'Voivodeships of Poland': "
              "Silesian / śląskie); WB name Slaskie."},
    {"alias_display": "Lower Silesian", "code": "POL001", "iso_a3": "POL",
     "basis": "English name of dolnośląskie (Wikipedia, 'Voivodeships of "
              "Poland': Lower Silesian / dolnośląskie)."},
    {"alias_display": "Lublin", "code": "POL004", "iso_a3": "POL",
     "basis": "English name of lubelskie (Wikipedia, 'Voivodeships of "
              "Poland': Lublin / lubelskie), named after its capital like the "
              "shipped aliases opole and lodz."},
    {"alias_display": "Lubelskie", "code": "POL004", "iso_a3": "POL",
     "basis": "Polish name of the voivodeship; the WB name misspells it "
              "Lubeiskie."},
    {"alias_display": "Lubusz", "code": "POL005", "iso_a3": "POL",
     "basis": "English name of lubuskie (Wikipedia, 'Voivodeships of "
              "Poland': Lubusz / lubuskie)."},
    {"alias_display": "Lesser Poland", "code": "POL006", "iso_a3": "POL",
     "basis": "English name of małopolskie (Wikipedia, 'Voivodeships of "
              "Poland': Lesser Poland / małopolskie)."},
    # NFKD leaves "ł" as it is, so the folded key is given explicitly.
    {"alias_display": "Małopolskie", "alias_key": "malopolskie",
     "code": "POL006", "iso_a3": "POL",
     "basis": "Polish name of the voivodeship; the WB name misspells it "
              "Malopolske."},
    {"alias_display": "Greater Poland", "code": "POL015", "iso_a3": "POL",
     "basis": "English name of wielkopolskie (Wikipedia, 'Voivodeships of "
              "Poland': Greater Poland / wielkopolskie)."},
    {"alias_display": "Zachodniopomorskie", "code": "POL016", "iso_a3": "POL",
     "basis": "Polish name of the voivodeship, written as one word; the WB "
              "name hyphenates it Zachodnio-Pomorskie."},
    {"alias_display": "North-West District", "code": "BWA007", "iso_a3": "BWA",
     "basis": "Botswana's district stored by the WB as Ngamiland (Wikipedia, "
              "'North-West District (Botswana)': 'The North-West District or "
              "Ngamiland'; Chobe separate since 2006)."},
    {"alias_display": "Barisal", "code": "BGD001", "iso_a3": "BGD",
     "basis": "English spelling of the division the WB stores as Barishal "
              "(Wikipedia article 'Barisal Division'); the counterpart of the "
              "shipped alias chittagong."},
    {"alias_display": "Arica and Parinacota", "code": "CHL004", "iso_a3": "CHL",
     "basis": "English name of the Región de Arica y Parinacota (Wikipedia, "
              "'Regions of Chile'); the WB name misspells it Arica y "
              "Painacota."},
    {"alias_display": "Arica y Parinacota", "code": "CHL004", "iso_a3": "CHL",
     "basis": "Spanish name of the region (Wikipedia, 'Regions of Chile'); "
              "the WB name misspells it Arica y Painacota."},
    {"alias_display": "Libertador General Bernardo O'Higgins", "code": "CHL008",
     "iso_a3": "CHL",
     "basis": "Official name of the Región del Libertador General Bernardo "
              "O'Higgins (Wikipedia, 'Regions of Chile'); the WB name "
              "abbreviates General to Gral."},
    {"alias_display": "Aysén del General Carlos Ibáñez del Campo",
     "code": "CHL001", "iso_a3": "CHL",
     "basis": "Official name of the Región Aysén del General Carlos Ibáñez "
              "del Campo (Wikipedia, 'Regions of Chile'); the WB name "
              "abbreviates General to Gral."},
    {"alias_display": "Magallanes and the Chilean Antarctica", "code": "CHL011",
     "iso_a3": "CHL",
     "basis": "English name of the region (Wikipedia, 'Magallanes Region': "
              "'Magallanes and the Chilean Antarctica Region in English')."},
    {"alias_display": "Magallanes y de la Antártica Chilena", "code": "CHL011",
     "iso_a3": "CHL",
     "basis": "Official Spanish name of the Región de Magallanes y de la "
              "Antártica Chilena (Wikipedia, 'Regions of Chile'); the WB "
              "name drops 'de la'."},
    {"alias_display": "Brussels", "code": "BEL001", "iso_a3": "BEL",
     "basis": "English name of the region (Wikipedia, 'Brussels': "
              "'officially the Brussels-Capital Region'); it had resolved as "
              "the start of the WB name until the single-word guard of "
              "2026-10-16."},
    # Swedish counties by name alone.  The WB names keep the genitive of
    # "X län" ("Hallands Laen") and spell out å, ä and ö, so "Halland" and
    # the unaccented "Skane" did not match.  Stockholm is left out: in
    # research text it is mostly the Stockholm Resilience Centre or
    # Convention, not the county.
    {"alias_display": "Halland", "code": "SWE005", "iso_a3": "SWE",
     "basis": "Halland County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Hallands Laen."},
    {"alias_display": "Kronoberg", "code": "SWE009", "iso_a3": "SWE",
     "basis": "Kronoberg County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Kronobergs Laen."},
    {"alias_display": "Jönköping", "code": "SWE007", "iso_a3": "SWE",
     "basis": "Jönköping County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Joenkoepings Laen."},
    {"alias_display": "Gävleborg", "code": "SWE003", "iso_a3": "SWE",
     "basis": "Gävleborg County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Gaevleborgs Laen."},
    {"alias_display": "Örebro", "code": "SWE011", "iso_a3": "SWE",
     "basis": "Örebro County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Oerebro Laen."},
    {"alias_display": "Skåne", "code": "SWE013", "iso_a3": "SWE",
     "basis": "Skåne County without 'County' (Wikipedia, 'Counties of "
              "Sweden'); WB name Skaane Laen."},
    # The Danish names of the five Danish regions ("Region Midtjylland").
    # The WB names are shorter English forms (Capital, Central Jutland, North
    # Jutland, South Denmark, Zealand); the regions' English names are
    # generated aliases.
    {"alias_display": "Region Hovedstaden", "code": "DNK001", "iso_a3": "DNK",
     "basis": "Danish name of the Capital Region of Denmark (Wikipedia, "
              "'Regions of Denmark'); WB name Capital."},
    {"alias_display": "Region Midtjylland", "code": "DNK002", "iso_a3": "DNK",
     "basis": "Danish name of the Central Denmark Region (Wikipedia, "
              "'Regions of Denmark'); WB name Central Jutland."},
    {"alias_display": "Region Nordjylland", "code": "DNK003", "iso_a3": "DNK",
     "basis": "Danish name of the North Denmark Region (Wikipedia, 'Regions "
              "of Denmark'); WB name North Jutland."},
    {"alias_display": "Region Syddanmark", "code": "DNK004", "iso_a3": "DNK",
     "basis": "Danish name of the Region of Southern Denmark (Wikipedia, "
              "'Regions of Denmark'); WB name South Denmark."},
    # NFKD leaves "æ" as it is, so the folded key is given explicitly.
    {"alias_display": "Region Sjælland", "alias_key": "region sjaelland",
     "code": "DNK005", "iso_a3": "DNK",
     "basis": "Danish name of Region Zealand (Wikipedia, 'Regions of "
              "Denmark'); WB name Zealand."},
    # Other official and English names the WB spells differently.
    {"alias_display": "Fryslân", "code": "NLD003", "iso_a3": "NLD",
     "basis": "Official name of the province, the West Frisian spelling the "
              "provincial council adopted in 1996 (Wikipedia, 'Friesland'); "
              "WB name Friesland."},
    {"alias_display": "Hamadan", "code": "IRN011", "iso_a3": "IRN",
     "basis": "English name of the province (Wikipedia, 'Provinces of "
              "Iran'); WB name Hamedan."},
    {"alias_display": "Khanty-Mansi Autonomous Okrug", "code": "RUS032",
     "iso_a3": "RUS",
     "basis": "English name of the okrug (Wikipedia, 'Khanty-Mansi "
              "Autonomous Okrug': Khanty-Mansi Autonomous Okrug–Yugra); WB "
              "name Khanty-mansyiskiy Okrug."},
)


def _fold(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )


def _normalize_key(raw: str) -> str:
    """N0: lowercase, strip, collapse spaces, NFKD-fold."""
    return _fold(raw.strip().lower())


def _is_country_name(key: str) -> bool:
    """G: the key is a country's ISO alpha-3 code, its name or a country
    alias itself, the matches ``resolve_country_code`` makes before its
    free-text step."""
    return resolve_country_code(key, substring=False) is not None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="build_data/raw_aliases.json")
    args = parser.parse_args(argv)

    raw_path = Path(args.raw)
    if not raw_path.is_file():
        sys.exit(f"ERROR: {raw_path} not found — run the generation workflow first")

    print(f"Loading raw aliases from {raw_path} ...")
    with open(raw_path, encoding="utf-8") as fh:
        raw_aliases: list[dict] = json.load(fh)
    print(f"  {len(raw_aliases)} raw candidates")
    raw_aliases += [
        {**alias, "source": "curated", "confidence": "high"}
        for alias in _CURATED_ALIASES
    ]
    print(f"  + {len(_CURATED_ALIASES)} curated aliases")

    # Load ADM1 data for validation
    print("Loading ADM1 metadata ...")
    _, _, _, adm1_metadata = _load_adm1_data()
    name_index = _get_adm1_name_index()
    valid_codes = set(adm1_metadata.keys())

    # Prevent circular contamination: any previously-shipped adm1_aliases.csv
    # would be loaded by _ensure_loaded() and used in Strategy 0 during R1
    # checks, making the R1 check self-referential.  Clear it so R1 tests the
    # pure name/substring resolver.
    import metacouplingllm.knowledge.adm1_pericoupling as _adm1_mod
    _adm1_mod._adm1_aliases = {}

    # ---------------------------------------------------------------------------
    # Per-candidate normalization + per-rule filtering
    # ---------------------------------------------------------------------------
    records: list[dict] = []
    drop_counts: dict[str, int] = defaultdict(int)

    for raw in raw_aliases:
        code = raw.get("code", "").strip()
        iso = raw.get("iso_a3", "").strip()
        alias_display = raw.get("alias_display", "").strip()
        alias_key_raw = raw.get("alias_key", "").strip()
        basis = raw.get("basis", "").strip()
        source = raw.get("source", "pretraining").strip()
        confidence = raw.get("confidence", "medium").strip()

        if code not in valid_codes:
            drop_counts["invalid_code"] += 1
            continue

        # N0
        key = _normalize_key(alias_key_raw or alias_display)
        if not key:
            drop_counts["N0_empty"] += 1
            continue

        # S1
        if len(key) < 4:
            drop_counts["S1_short"] += 1
            continue

        # G
        if _is_country_name(key):
            drop_counts["G_country_name"] += 1
            continue

        # W1
        if key in _W1_WORDS:
            drop_counts["W1_generic"] += 1
            continue

        # DR: manual review denylist (semantic scope violations the
        # deterministic rules cannot detect — see _REVIEW_DENYLIST)
        if (key, code) in _REVIEW_DENYLIST:
            drop_counts["DR_review_denylist"] += 1
            continue

        # R1 / O1.  A one-word key that the matcher found without a hint only
        # as part of a longer name ("beijing" in "Beijing Shi") is not
        # redundant: its single-word guard now refuses that, so without a
        # hint the alias is what resolves the word.
        resolved = _resolve_adm1_name(key, iso)
        needed_without_hint = (
            _resolve_adm1_name(key, None) != code
            and _resolve_adm1_name(key, None, lone_word_substring=True) == code
        )
        if resolved == code and not needed_without_hint:
            drop_counts["R1_redundant"] += 1
            continue
        override = False
        if resolved is not None and resolved != code:
            override = True

        # X1: fold-aware no-hint shadow.  Strategy 0 runs FIRST and, with no
        # country hint, would shadow another region's natural resolution.  Drop
        # any alias whose key — resolved with NO hint (aliases cleared above) —
        # lands on a DIFFERENT code.  Catches folded cross-country collisions
        # that U3 misses because U3 only consults the accented name index
        # (e.g. folded "ash sharqiyah" → SAU008 while the alias claims OMN002).
        nohint = _resolve_adm1_name(key, None)
        if nohint is not None and nohint != code:
            drop_counts["X1_nohint_shadow"] += 1
            continue

        records.append({
            "code": code,
            "iso_a3": iso,
            "alias_display": alias_display,
            "alias_key": key,
            "basis": basis,
            "source": source,
            "confidence": confidence,
            "override": override,
        })

    print(f"  After N0/G/S1/W1/DR/R1/O1/X1: {len(records)} candidates remain")
    for rule, count in sorted(drop_counts.items()):
        print(f"    dropped by {rule}: {count}")

    # ---------------------------------------------------------------------------
    # U1: intra-country key uniqueness
    # ---------------------------------------------------------------------------
    # key × country → list of codes
    intra: dict[tuple[str, str], list[str]] = defaultdict(list)
    for rec in records:
        intra[(rec["alias_key"], rec["iso_a3"])].append(rec["code"])

    u1_collisions: set[tuple[str, str]] = {k for k, v in intra.items() if len(set(v)) > 1}
    before_u1 = len(records)
    records = [r for r in records if (r["alias_key"], r["iso_a3"]) not in u1_collisions]
    drop_counts["U1_intra"] = before_u1 - len(records)
    print(f"  After U1 (intra-country uniqueness): {len(records)}")

    # ---------------------------------------------------------------------------
    # U2: cross-country key uniqueness
    # ---------------------------------------------------------------------------
    cross: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        cross[rec["alias_key"]].add(rec["iso_a3"])

    u2_ambiguous: set[str] = {k for k, isos in cross.items() if len(isos) > 1}
    before_u2 = len(records)
    records = [r for r in records if r["alias_key"] not in u2_ambiguous]
    drop_counts["U2_cross"] = before_u2 - len(records)
    print(f"  After U2 (cross-country uniqueness): {len(records)}")

    # ---------------------------------------------------------------------------
    # U3: don't shadow an existing canonical name for a different code
    # ---------------------------------------------------------------------------
    before_u3 = len(records)
    kept_u3 = []
    for rec in records:
        existing = name_index.get(rec["alias_key"])
        if existing:
            existing_codes = {c for c, _ in existing}
            if existing_codes != {rec["code"]}:
                drop_counts["U3_shadows"] += 1
                continue
        kept_u3.append(rec)
    records = kept_u3
    drop_counts.setdefault("U3_shadows", 0)
    print(f"  After U3 (no shadow): {len(records)}")

    # ---------------------------------------------------------------------------
    # D0: deduplicate identical (code, alias_key) pairs produced by parallel
    # agents.  A curated record wins over a generated one with the same pair,
    # so the audit table keeps the source its basis cites.
    # ---------------------------------------------------------------------------
    by_pair: dict[tuple[str, str], dict] = {}
    for rec in records:
        pair = (rec["code"], rec["alias_key"])
        kept = by_pair.get(pair)
        if kept is None or (
            rec["source"] == "curated" and kept["source"] != "curated"
        ):
            by_pair[pair] = rec
    deduped = list(by_pair.values())
    drop_counts["D0_duplicate"] = len(records) - len(deduped)
    records = deduped
    print(f"  After D0 (dedup same code+key): {len(records)}")

    # ---------------------------------------------------------------------------
    # C1: cap at 3 generated aliases per code (high > medium; pretraining >
    # uncertain; alphabetical)
    # ---------------------------------------------------------------------------
    def _sort_key(r: dict) -> tuple:
        conf_order = {"high": 0, "medium": 1}.get(r["confidence"], 2)
        src_order = {"pretraining": 0, "uncertain": 1}.get(r["source"], 2)
        return (conf_order, src_order, r["alias_display"])

    by_code: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_code[rec["code"]].append(rec)

    final_shipped: list[dict] = []
    final_audit: list[dict] = []
    for code_key, recs in by_code.items():
        recs_sorted = sorted(recs, key=_sort_key)
        # Split overrides out (audit-only, never shipped).  Curated aliases
        # were checked by hand and always ship; the cap limits the
        # generated ones.
        non_override = [r for r in recs_sorted if not r["override"]]
        shipped = [r for r in non_override if r["source"] == "curated"] + [
            r for r in non_override if r["source"] != "curated"
        ][:3]
        final_shipped.extend(shipped)
        # Audit includes everything (overrides flagged)
        final_audit.extend(recs_sorted)

    print(f"\nFinal shipped aliases: {len(final_shipped)}")
    override_count = sum(1 for r in final_audit if r["override"])
    print(f"Override candidates (excluded from shipped, in audit): {override_count}")
    print(f"Codes with at least one alias: {len(by_code)}")

    # Every curated alias was checked by hand: if a rule dropped one, stop
    # before writing anything.
    shipped_pairs = {(r["alias_key"], r["code"]) for r in final_shipped}
    dropped_curated = []
    for alias in _CURATED_ALIASES:
        key = _normalize_key(alias.get("alias_key") or alias["alias_display"])
        if (key, alias["code"]) not in shipped_pairs:
            dropped_curated.append(f"{key} -> {alias['code']}")
    if dropped_curated:
        sys.exit(
            "ERROR: validation dropped curated aliases: "
            + "; ".join(dropped_curated)
        )

    # ---------------------------------------------------------------------------
    # Write shipped CSV
    # ---------------------------------------------------------------------------
    shipped_path = Path("src/metacouplingllm/data/adm1_aliases.csv")
    shipped_path.parent.mkdir(parents=True, exist_ok=True)
    shipped_fields = ["alias_key", "alias_display", "code", "iso_a3"]
    with open(shipped_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=shipped_fields)
        w.writeheader()
        for rec in sorted(final_shipped, key=lambda r: (r["code"], r["alias_key"])):
            w.writerow({f: rec[f] for f in shipped_fields})
    print(f"Written: {shipped_path}")

    # ---------------------------------------------------------------------------
    # Write audit CSV
    # ---------------------------------------------------------------------------
    audit_path = Path("docs/adm1_aliases_audit.csv")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_fields = ["alias_key", "alias_display", "code", "iso_a3", "basis", "source", "confidence", "override"]
    with open(audit_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=audit_fields)
        w.writeheader()
        for rec in sorted(final_audit, key=lambda r: (r["code"], r["alias_key"])):
            w.writerow({f: rec.get(f, "") for f in audit_fields})
    print(f"Written: {audit_path}")

    # ---------------------------------------------------------------------------
    # Final assertions
    # ---------------------------------------------------------------------------
    print("\nRunning final assertions ...")
    shipped_keys = [r["alias_key"] for r in final_shipped]
    assert len(shipped_keys) == len(set(shipped_keys)), "FAIL: duplicate alias_key in shipped CSV"
    for rec in final_shipped:
        assert rec["code"] in valid_codes, f"FAIL: unknown code {rec['code']}"
        assert not _is_country_name(rec["alias_key"]), f"FAIL: alias_key is a country name: {rec['alias_key']}"
    print("All assertions passed.")

    # Summary by drop rule
    print("\nDrop summary:")
    total_dropped = len(raw_aliases) - len(final_shipped)
    for rule, count in sorted(drop_counts.items()):
        print(f"  {rule}: {count}")
    print(f"  Total dropped: {total_dropped} / {len(raw_aliases)}")


if __name__ == "__main__":
    main()
