# 10 — Pipeline metadata

## Map render

| Key | Value |
|---|---|
| `last_map_type` | country |
| `last_map_notice` | None |
| `result.map_present` | True |

## Token usage (per LLM call)

| Call | Label | Input | Output | Total | Duration (s) |
|---|---|---|---|---|---|
| 1 | main_analysis | 27682 | 17158 | 44840 | 422.25 |
| 2 | map_extraction | 12342 | 4260 | 16602 | 79.36 |
| 3 | abstract | 12798 | 5065 | 17863 | 102.51 |
| **TOTAL** | — | **52822** | **26483** | **79305** | — |

## Wall-clock breakdown

| Key | Value |
|---|---|
| `total_wall_clock_s` | 1516.32 |
| `sum_of_llm_call_s` | 604.12 |
| `other_s (uncaptured web calls, retrieval, render, parsing)` | 912.2 |
