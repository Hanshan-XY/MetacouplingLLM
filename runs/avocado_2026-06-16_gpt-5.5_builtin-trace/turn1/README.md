# Pipeline trace

**Query:** Impact of avocado production and trade in Mexico on sustainability

**Model:** `gpt-5.5`

**Wall-clock:** 5263.9s | **LLM calls:** 3 | **Total tokens:** 74537

## Reading this trace

- Files are numbered by **pipeline stage** (the order steps run), not by model-call order. For example, the structured web-extraction call appears early, at `02`, because it runs during the web-search stage, before the main analysis.
- The **LLM calls** count above, and the token table in `10_pipeline_metadata.md`, include only calls captured through the assistant's `chat()` proxy. The structured web-extraction call (`02`, when present) is *summarized rather than chat-captured*, because it is issued by the provider's native web-search client rather than that proxy; its raw inputs and outputs are in `01` and `03`.
- Not every file appears in every run: the web, RAG, map, and supplementary stages are written only when the corresponding feature is enabled.

## Artifacts

| File | Description |
|---|---|
| [`00_run_config.md`](./00_run_config.md) | Query, model, parameters, git SHA, and environment. |
| [`01_web_results_raw.md`](./01_web_results_raw.md) | Raw results returned by the web search. |
| [`02_llm_call_web_extraction.md`](./02_llm_call_web_extraction.md) | Structured web-extraction model call (summarized; see above). |
| [`03_web_structured_signals.md`](./03_web_structured_signals.md) | Structured signals and evidence cards from the web results. |
| [`04_rag_chunks.md`](./04_rag_chunks.md) | Literature passages retrieved from the RAG corpus. |
| [`05_llm_call_main_analysis.md`](./05_llm_call_main_analysis.md) | Main framework-analysis model call, with the full seven-layer system prompt. |
| [`06_parsed_analysis.md`](./06_parsed_analysis.md) | Parsed ParsedAnalysis structure (parser output of the main response). |
| [`07_llm_call_map_extraction.md`](./07_llm_call_map_extraction.md) | Map-signal extraction model call. |
| [`08_map_data.md`](./08_map_data.md) | Structured map data used to render the figure. |
| [`09_formatted_output.md`](./09_formatted_output.md) | Final formatted report. |
| [`10_pipeline_metadata.md`](./10_pipeline_metadata.md) | Per-call token usage, wall-clock breakdown, and map metadata. |
| [`map.png`](./map.png) | Rendered metacoupling map. |
