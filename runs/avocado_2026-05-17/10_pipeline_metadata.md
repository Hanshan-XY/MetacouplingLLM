# 10 — Pipeline metadata

## Map render

| Key | Value |
|---|---|
| `last_map_type` | None |
| `last_map_notice` |   --- Auto-map did not generate a figure. It currently supports only countries and ADM1 subnational regions, not city, watershed, protected-area, reserve, or park geometries. Specify the parent ADM1 r... |
| `result.map_notice` |   --- Auto-map did not generate a figure. It currently supports only countries and ADM1 subnational regions, not city, watershed, protected-area, reserve, or park geometries. Specify the parent ADM1 r... |
| `result.map_present` | False |

## Flow parse warnings (legacy regex path)

_(none)_

## Citation accounting

| Key | Value |
|---|---|
| `turn_number` | 1 |
| `turn_passage_counts` | {1: 8} |
| `turn_web_counts` | {1: 5} |

## Token usage (per LLM call)

| Call | Purpose | Input | Output | Total | Duration (s) |
|---|---|---|---|---|---|
| 1 | web extraction | 1022 | 646 | 1668 | 12.73 |
| 2 | main analysis | 13372 | 8353 | 21725 | 153.21 |
| 3 | map extraction | 3377 | 607 | 3984 | 11.04 |
| **TOTAL** | — | **17771** | **9606** | **27377** | — |

## Wall-clock breakdown

| Key | Value |
|---|---|
| `total_wall_clock_s` | 322.99 |
| `sum_of_llm_call_s` | 176.98 |
| `non_llm_s (retrieval + render + parsing)` | 146.01 |
