# Specification: the de facto administrators of the disputed-area tracts (campaign `da1`)

Status: **APPROVED, 2026-09-25**, by the maintainer's rulings recorded verbatim in section 6. The rule is applied once, in
`scripts/build_pericoupling_db.py`; only implementation defects are corrected after this point, each logged in section 7.

## 1. Origin

The manuscript said that three of the 24 disputed-area tracts are left unassigned because they "are islands with no land
border, where an assignment would restore no connectivity". The maintainer asked whether the islands really lack a single
de facto administrator and why a unified rule was not used ("Why not assigned them? I know assigning them won't increase
adjacent pairs, but why not using unified rules?"). The islands had been left out only because the build's check requires
an administrator to touch one of its tracts. Checking the per-tract alternative showed that the Karakoram Range tract,
assigned to Pakistan, is by location (75.9-77.5 E, 35.5-36.6 N), size (4,861 km2), drainage (every channel inside drains to
the Yarkand River) and its K2 edge the Shaksgam Valley (Trans-Karakoram Tract). The maintainer: "assigning the 3 island and
look into the Karakoram tract."

## 2. Sources

- **World Bank**, NDLSA layer of the pinned 2026-05-14 release: no administrator for any tract (`SOVEREIGN` empty for all 24,
  `WB_STATUS` "Non-determined legal status area").
- **Natural Earth v5.1.2** (the latest release, 2022-05-13), `ne_10m_admin_0_disputed_areas` and `ne_10m_admin_0_countries`,
  downloaded 2026-09-25 from naciscdn.org (maintainer: "Yes, and make sure to use the latest version."; "Agree all."). The
  `.shp` and `.dbf` of both layers are byte-identical to the v5.1.2 release files on GitHub (git blob hashes: disputed areas
  shp `44764cb4`, dbf `ceb7947a`; countries shp `a10056b7`, dbf `0d66afaf`); neither layer changed after April 2022, which is
  why the files' version note reads 5.1.1. Local copies: `build_data/naturalearth/`.
- OpenStreetMap was consulted for the Karakoram tract only (three points inside it lie in Xinjiang, China); it is not a
  source of the rule.

## 3. The rule

1. **Administrator.** Each World Bank tract is assigned whole to the administrator Natural Earth records for the largest part
   of it: Natural Earth's disputed-area records first ("Admin. by ..."), its country polygons for the rest of the tract. A
   part outside every Natural Earth polygon (sea, or where the two coastlines differ) carries no information and is left out
   of the count. A dependency counts for its sovereign (British Indian Ocean Territory, the Falklands and South Georgia for
   the United Kingdom).
2. **Unassigned.** A tract whose largest part has no administrator the World Bank lists as a country stays unassigned.
3. **No exceptions** (maintainer: "Remove the exception (pure majority)"). Natural Earth's own smaller areas inside a tract do
   not override the majority: Western Sahara goes to Morocco whole, although Natural Earth records its eastern third
   (33.5%) as self-administered by the Sahrawi Republic; Jammu and Kashmir goes to India whole, although Natural Earth
   records the Siachen Glacier (1.9%) with claims only; the Ilemi Triangle goes to Kenya whole, although Natural Earth
   records a 17 km2 area inside it (0.5%) as South Sudanese.
4. **The check.** Every assigned tract must touch its administrator's territory (the administrator's own polygon or another
   tract assigned to it), unless the tract touches no unit at all (the three islands, which then add no pair). This replaces
   the check that one tract per administrator touches it.
5. **Restored country border.** Measured between the two countries' de facto territories (each country's polygon with the
   tracts assigned to it), not against the other country's standard polygon. Pair detection is unchanged (a pair is restored
   only where the two countries do not touch in the standard layer).
6. **Overlay row label.** Unchanged: a country-level row lists all the tracts of its administrator (maintainer: "All the
   administrator's tracts").
7. **Provinces.** The authored province lists stay where a tract keeps its administrator. For a tract with a new
   administrator, the list is that country's World Bank provinces that touch the tract.

## 4. The tracts under the rule (shares of the World Bank tract's area; `build_data/ndlsa_ne/ne_admin_of_tracts.csv`)

| Tract | Build before | Natural Earth: largest administrator (share of the tract; record) | Rest of the tract | Assigned |
|---|---|---|---|---|
| Aksai Chin | China | China 99.0%; Aksai Chin (Admin. by China; Claimed by India) | India 1.0% | **China** |
| Kauirik | China | China 57.0%; China [CHN] | India 43.0% | **China** |
| Lapthal | China | India 83.5%; India [IND] | China 16.5% | **India** |
| Shipki Pass | China | India 54.5%; India [IND] | China 45.5% | **India** |
| Chumar East | India | India 100.0%; Jammu and Kashmir (Admin. by India; Claimed by Pakistan) | - | **India** |
| Chumar West | India | India 100.0%; Jammu and Kashmir (Admin. by India; Claimed by Pakistan) | - | **India** |
| Demchok | India | India 97.2%; Demchok (Admin. by India; Claimed by China) | China 2.8% | **India** |
| Jadh Ganga Valley | India | India 95.0%; Tirpani Valleys (Admin. by India; Claimed by China) | China 5.0% | **India** |
| Arunachal Pradesh | India | India 96.5%; Arunachal Pradesh (Admin. by India; Claimed by China) | China 3.2% | **India** |
| Jammu and Kashmir | India | India 97.1%; Jammu and Kashmir (Admin. by India; Claimed by Pakistan) | no administrator recorded 1.9%, Pakistan 0.5% | **India** |
| Kalapani | India | India 74.7%; India [IND] | China 22.5%, Nepal 2.9% | **India** |
| Doklam | Bhutan | Bhutan 99.8%; Bhutan [BTN] | - | **Bhutan** |
| Gilgit Baltistan | Pakistan | Pakistan 97.6%; Gilgit-Baltistan (Admin. by Pakistan; Claimed by India) | India 1.1%, China 1.0% | **Pakistan** |
| Karakoram Range | Pakistan | China 99.9%; Shaksam Valley (Admin. by China; Ceded to China by Pakistan; Claimed by India) | - | **China** |
| Golan Heights | Israel | Israel 96.1%; Golan Heights (Admin. By Israel; Claimed by Syria) | Syria 3.2%, Jordan 0.7% | **Israel** |
| Shebaa Farms Dispute | Israel | Israel 95.5%; Golan Heights (Admin. By Israel; Claimed by Syria) | Lebanon 4.5% | **Israel** |
| No Man's Land | unassigned | Israel 94.2%; No Man's Land (Fort Latrun) (Admin. By Israel; Claimed by Palestine) | Palestine 5.8% | **Israel** |
| Western Sahara | Morocco | Morocco 65.9%; W. Sahara (Admin. by Morocco; Claimed by Western Sahara) | no administrator recorded 33.5% | **Morocco** |
| Ilemi Triangle | Kenya | Kenya 98.9%; Ilemi Triangle (Admin. by Kenya; Claimed by South Sudan) | South Sudan 1.1% | **Kenya** |
| Abyei | unassigned | Sudan 100.0%; Abyei (Admin. by Sudan; Claimed by South Sudan) | - | **Sudan** |
| UN Buffer Zone | unassigned | no administrator recorded 89.3%; Cyprus U.N. Buffer Zone (Cyprus No Mans Area) | Cyprus 8.9%, United Kingdom 1.6% | **unassigned** |
| British Indian Ocean Territory | unassigned | United Kingdom 50.2%; Diego Garcia NSF (Leased to U.S.A by U.K.; Claimed by Mauritius and Seychelles) | outside every Natural Earth polygon 49.8% | **United Kingdom** |
| South Georgia and South Sandwich Islands | unassigned | United Kingdom 92.1%; S. Georgia (Admin. by U.K.; Claimed by Argentina) | outside every Natural Earth polygon 7.9% | **United Kingdom** |
| Falkland Islands | unassigned | United Kingdom 89.2%; Falkland Is. (Admin. by U.K.; Claimed by Argentina) | outside every Natural Earth polygon 10.8% | **United Kingdom** |

Changed: Karakoram Range (Pakistan -> China), Lapthal and Shipki Pass (China -> India), No Man's Land (none -> Israel), Abyei
(none -> Sudan), the three islands (none -> United Kingdom). Kauirik stays with China by majority (57.0%): Natural Earth's
"Samdu Valleys" record (Admin. by India) covers 42.1% of the tract (maintainer: "I agree with the the majority rule,
including Kauirik staying with China"). Only the UN Buffer Zone remains unassigned.

New province lists: Lapthal IND035 (Uttarakhand); Shipki Pass IND014 (Himachal Pradesh); Karakoram Range CHN028 (Xinjiang);
No Man's Land ISR001, ISR003; Abyei SDN012, SDN013; the three islands none (country level only).

## 5. Expected effect (measured read-only with the build's own derivation, `build_data/ndlsa_ne/measure_ne_overlay.py`)

- Country level: the same 3 pairs. China-Pakistan: tracts "Gilgit Baltistan", restored border 491.2 -> 455.1 km (between
  de facto territories). Israel-Syria: 79.1 km, label gains "No Man's Land". Morocco-Mauritania: unchanged (1,543.6 km).
- Province level: 13 -> 16 pairs: + Jerusalem (ISR003) - Ramallah (PSE013), 12.4 km, across the Latrun no-man's-land;
  + Southern Darfur (SDN012) - Warrap (SSD010), 47.8 km, and + Southern Kordofan (SDN013) - Warrap, 34.2 km, across Abyei.
- ADM1 edges 8,458 -> 8,461; moderate 8,062 -> 8,065; stringent 7,655 -> 7,658; the strict view (`de_facto_borders=False`)
  unchanged at 8,445; the ADM0 matrix unchanged (326 / 320 / 300).
The full rebuild (section 7) confirms or corrects these figures.

## 6. Maintainer rulings (2026-09-25, verbatim)

- "Why not assigned them? I know assigning them won't increase adjacent pairs, but why not using unified rules?"
- "assigning the 3 island and look into the Karakoram tract."
- "Yes, and make sure to use the latest version." (download of the Natural Earth disputed-areas layer)
- "Agree all." (Natural Earth's recorded administrator as the rule; download of the countries layer)
- On the measured changes: "Adopt all (Recommended)"; on the border length: "Between de facto territories (Recommended)";
  on the row label: "All the administrator's tracts".
- "I agree with the the majority rule, including Kauirik staying with China"
- "Remove the exception (pure majority)"

## 7. Change log

1. 2026-09-25, full rebuild from the pinned GeoPackages (`python scripts/build_all.py --full`, log `da1_full_rebuild.log`):
   the overlay derivation gives 3 country pairs and 16 province pairs; 12/12 headline counts (ADM1 8,461, moderate 8,065,
   stringent 7,658, water-only 803 = 407 / 396, ADM0 326 / 320 / 300). Against the shipped files
   (`da1_full_rebuild_compare.txt`): `disputed_overlay_pairs.csv` + the three province rows, the Israel-Syria label, the
   China-Pakistan label and 455.1 km; `pericoupled_adm1_edge_list.csv` + the three rows; every other row and its order
   unchanged; `PeriTelecoupling_clean.csv` and `water_separated_pairs.csv` byte-identical. Section 5 confirmed as measured.
2. 2026-09-25, found while updating the documents, not a defect of the rule: the corridor census takes only pairs that are
   not edges, so the three new province pairs leave its population (23,410 -> 23,407), and Jerusalem-Ramallah, which it had
   nominated (short-gap presence rule, a 422 m gap) and the nt2 two-pass had rejected as a dry gap, leaves its nominations
   (243 -> 242). The census computes each pair on its own, so dropping the three pairs from its record
   (`build_data/geodesic_distances/census_gd1.csv`) gives what a re-run on the new edge list gives; the manuscript guard
   (`paper/verify_v3_section3.py`) now does so.
3. 2026-09-25, implementation defect: a comment in `_NDLSA_TRACT_ADMIN` gave Lapthal as India 84%; Natural Earth's share
   is 83.5% (83%). Comment corrected; no effect on the data.
4. 2026-09-25, `docs/ndlsa_tract_audit.csv` regenerated by `make_tract_audit.py` from the build's geometry, the new
   overlay and `ne_admin_of_tracts.csv` (Natural Earth's administrator share, record and outside share per tract). Its
   country-level status now says which tracts restore a pair (Shebaa Farms touches no Syrian territory, so it restores
   none; the earlier table credited every tract on the row's label) and which lie on a restored border (Karakoram Range,
   on the China-Pakistan border).
5. 2026-09-25, older comments corrected: `_NDLSA_TRACT_ADMIN` marked Shebaa Farms as producing the Israel-Syria pair and
   `_NDLSA_TRACT_ADM1` listed it among the tracts that produce province pairs. The tract touches Lebanon (20.9 km), Israel
   (4.4 km) and the Golan tract (10.8 km) and lies about 4 km (0.038 degrees) from Syria's polygon, so it produces neither
   (the Israel-Syria pairs come from the Golan). Comments only; METHODS' list of tracts that change no pair now includes it.
