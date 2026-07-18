# Metacoupling Release Checklist

Use this list before publishing a new release to PyPI or GitHub.

## 1. Legal and Data Review

- Confirm you have the right to redistribute every file in `src/metacouplingllm/data/`.
- Re-check whether `src/metacouplingllm/data/Papers.zip` can be shipped publicly.
- Confirm any required attribution for geographic datasets is present in the docs.

## 2. Version and Metadata

- Update the version in `pyproject.toml`.
- Confirm `metacouplingllm.__version__` reflects the new pyproject.toml version - it is resolved dynamically, no manual edit needed.
- Make sure README, INTRODUCTION, and MANUAL examples still match the current API.
- Review package metadata such as authors, homepage, and documentation links.

## 3. Clean Build Inputs

- Remove generated caches and temporary folders.
- Make sure backup files are excluded from the distribution.
- Check that only intended package data is included in `src/metacouplingllm/data/`.

## 3b. Rebuild RAG Embeddings (when corpus or chunker changes)

The package ships pre-computed BGE-base (BAAI/bge-base-en-v1.5) embeddings for the bundled
RAG corpus in `src/metacouplingllm/data/chunk_embeddings.npy`. This file
is consumed at runtime by `EmbeddingRetriever` and must stay in sync
with `Papers.zip` and the chunker output. Rebuild when any of these
change:

- `src/metacouplingllm/data/Papers.zip` (new / updated papers)
- The chunking parameters or output order in `rag._chunk_markdown`
- The default embedding model (`DEFAULT_EMBEDDING_MODEL` in `rag.py`)

To rebuild::

    pip install fastembed
    python scripts/build_embeddings.py --force

The script writes `chunk_embeddings.npy` (~15 MB for BGE-base) **and a
sidecar `chunk_embeddings.manifest.json`** (chunk-order fingerprint). Commit
BOTH files so they ship together — a missing or stale manifest makes
`EmbeddingRetriever` raise `RuntimeError` ("Chunk-order fingerprint
mismatch") at runtime for every RAG user.

## 4. Test and Smoke Test

- Run `python -m pytest`.
- Run targeted tests for optional features if you changed RAG, web search, or maps.
- Create a fresh virtual environment and test a minimal install:
  `pip install metacouplingllm`
- Test at least one extra you expect most users to need:
  `pip install "metacouplingllm[openai]"`

## 5. Build the Distribution

- Install the build tool if needed:
  `python -m pip install build`
- Build both artifacts:
  `python -m build`
- Inspect the wheel contents:
  `python -m zipfile -l dist/<wheel-name>.whl`

## 6. Pre-Publish Validation

- Verify the long description renders correctly from `INTRODUCTION.md`.
- Check that `LICENSE` is included.
- Confirm `README.md`, `INTRODUCTION.md`, and package examples use valid install commands.
- Decide whether to publish to TestPyPI first.

## 7. Post-Publish Checks

- Install the published version in a clean environment.
- Run one end-to-end example from the docs.
- Open the PyPI project page and confirm the rendered description and links look right.
