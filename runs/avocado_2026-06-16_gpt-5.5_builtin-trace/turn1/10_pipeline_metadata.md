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
| 1 | main_analysis | 24484 | 11380 | 35864 | 222.81 |
| 2 | map_extraction | 11503 | 3403 | 14906 | 73.02 |
| 3 | abstract | 23161 | 606 | 23767 | 15.96 |
| **TOTAL** | — | **59148** | **15389** | **74537** | — |

## Wall-clock breakdown

| Key | Value |
|---|---|
| `total_wall_clock_s` | 5263.93 |
| `sum_of_llm_call_s` | 311.79 |
| `non_llm_s (retrieval + render + parsing)` | 4952.14 |
