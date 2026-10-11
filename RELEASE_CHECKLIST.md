# MetacouplingLLM Release Checklist

Use this list before publishing a new release to PyPI or GitHub.

## 1. Legal and Data Review

- Confirm you have the right to redistribute every file in `src/metacouplingllm/data/`.
- Re-check whether `src/metacouplingllm/data/Papers.zip` can be shipped publicly.
- Confirm any required attribution for geographic datasets is present in the docs.

## 2. Version and Metadata

- Update the version in `pyproject.toml`. Review the `Development Status` classifier there; change it only when the project's maturity changes, not at every release.
- Update the last-resort version literal in `_resolve_version()` (`src/metacouplingllm/__init__.py`, the `else "x.y.z"` branch) to the same version.
- In `CHANGELOG.md`, turn `[Unreleased]` into a dated `[x.y.z]` section and leave a new empty `[Unreleased]` above it.
- Update the version stamps in `INTRODUCTION.md` and `MANUAL.md` (find them with `grep -n "<old version>" INTRODUCTION.md MANUAL.md`).
- Reinstall (`pip install -e .`), then confirm `metacouplingllm.__version__` shows the new version: installed metadata is read before `pyproject.toml`, so an install made at the old version keeps reporting it.
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
- The paper key any chunk carries: the bibliography
  (`telecoupling_literature.bib`) or the filename-to-entry matching in
  `rag.py` (the fingerprint covers each chunk's paper key)
- The default embedding model (`DEFAULT_EMBEDDING_MODEL` in `rag.py`)

To rebuild::

    pip install fastembed
    python scripts/build_embeddings.py --force

`scripts/build_embeddings.py` does not read `DEFAULT_EMBEDDING_MODEL`; its
`--model` default is `BAAI/bge-base-en-v1.5`. After changing the constant,
pass the new model explicitly, or the corpus is encoded with the old model and
the loader cannot tell (it never compares the manifest's `embedding_model`)::

    python scripts/build_embeddings.py --force --model <new model>

The script writes `chunk_embeddings.npy` (~15 MB for BGE-base) **and a
sidecar `chunk_embeddings.manifest.json`** (chunk-order fingerprint). Commit
BOTH files so they ship together. A stale manifest (its fingerprint differs
from the chunks `RAGEngine.load()` builds) makes `EmbeddingRetriever` raise
`RuntimeError` ("Chunk-order fingerprint mismatch"), but `RAGEngine` catches
it and, with the default `rag_backend="auto"`, silently falls back to TF-IDF,
so every RAG user would get weaker retrieval and no error. A missing
manifest is not an error either: the check is skipped (a warning is logged
only with `verbose=True`). CI skips `tests/test_rag.py`, which holds the
manifest-vs-chunks check, so before a release run
`python -m pytest tests/test_rag.py::TestBundledManifestIntegrity` and confirm
that a RAG run prints `[RAG] Using embedding retrieval backend`.

## 4. Test

- Run `python -m pytest`.
- Run targeted tests for optional features if you changed RAG, web search, or maps.

## 5. Build the Distribution

- Install the build tool if needed:
  `python -m pip install build`
- Build both artifacts:
  `python -m build`
- Inspect the wheel contents:
  `python -m zipfile -l dist/<wheel-name>.whl`

## 6. Pre-Publish Validation

- Smoke-test the built wheel, not the PyPI release (which is still the
  previous version until you publish). Create a fresh virtual environment
  outside the repository (for example `python -m venv ../metacouplingllm-smoke`)
  and test a minimal install:
  `pip install dist/metacouplingllm-<version>-py3-none-any.whl`
- Test at least one extra you expect most users to need:
  `pip install "dist/metacouplingllm-<version>-py3-none-any.whl[openai]"`
- Verify the long description renders correctly from `INTRODUCTION.md`.
- Check that `LICENSE` is included.
- Confirm `README.md`, `INTRODUCTION.md`, and package examples use valid install commands.
- Decide whether to publish to TestPyPI first.

## 7. Post-Publish Checks

- Install the published version in a clean environment.
- Run one end-to-end example from the docs.
- Open the PyPI project page and confirm the rendered description and links look right.
