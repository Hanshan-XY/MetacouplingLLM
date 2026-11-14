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
| 1 | main_analysis | 27761 | 19232 | 46993 | 502.62 |
| 2 | map_extraction | 11058 | 7271 | 18329 | 149.49 |
| 3 | abstract | 12061 | 4161 | 16222 | 73.99 |
| **TOTAL** | — | **50880** | **30664** | **81544** | — |

## Wall-clock breakdown

| Key | Value |
|---|---|
| `total_wall_clock_s` | 1708.52 |
| `sum_of_llm_call_s` | 726.1 |
| `other_s (uncaptured web calls, retrieval, render, parsing)` | 982.42 |
