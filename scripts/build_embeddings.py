#!/usr/bin/env python3
"""Pre-compute corpus embeddings for the RAG engine.

Runs the fastembed encoder over every chunk in the bundled RAG corpus
(``Papers.zip`` → 9,581 chunks as of writing) and saves the resulting
(n_chunks, dims) numpy array to
``src/metacoupling/data/chunk_embeddings.npy``.

This file is bundled with the package so users never have to re-encode
the corpus. The build script should be re-run whenever:

- The bundled papers change (new papers added / chunking parameters
  tweaked).
- The chunker output order changes.
- The embedding model changes (e.g., BGE-small → BGE-base).

Usage::

    python scripts/build_embeddings.py

Options::

    --model MODEL_NAME   Model to use (default: BAAI/bge-base-en-v1.5)
    --papers-dir PATH    Markdown corpus directory to encode
    --force              Overwrite an existing .npy file without prompting

Requirements::

    pip install fastembed
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
import time
from pathlib import Path

# Make the package importable when run from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-compute corpus embeddings for the metacoupling RAG engine.",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="fastembed model name (default: BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--papers-dir",
        default=None,
        help="Directory of markdown papers to encode. Defaults to bundled Papers.zip.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32"),
        default="float16",
        help=(
            "Stored vector dtype (default: float16). float16 keeps the "
            "distributed package smaller; runtime scoring converts "
            "vectors back to float32."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing embedding file without prompting.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the .npy file. Defaults to "
             "src/metacoupling/data/chunk_embeddings.npy",
    )
    args = parser.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("ERROR: numpy is not installed. Run `pip install numpy`.")
        return 1

    try:
        from fastembed import TextEmbedding
    except ImportError:
        print(
            "ERROR: fastembed is not installed. "
            "Run `pip install fastembed`."
        )
        return 1

    from metacouplingllm.knowledge.rag import (
        DEFAULT_EMBEDDING_DIMS,
        RAGEngine,
        _extract_bundled_papers,
        compute_chunk_fingerprint,
    )

    output_path = Path(args.output) if args.output else (
        ROOT / "src" / "metacoupling" / "data" / "chunk_embeddings.npy"
    )

    if output_path.exists() and not args.force:
        reply = input(
            f"{output_path} already exists. Overwrite? [y/N] "
        ).strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return 1

    print("=" * 60)
    print("METACOUPLING RAG EMBEDDING BUILDER")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  Output:  {output_path}")
    print(f"  Dtype:   {args.dtype}")
    print()

    # --- Load chunks via the existing RAG engine ---
    print("[1/3] Loading papers and chunking...")
    papers_dir = Path(args.papers_dir) if args.papers_dir else (
        _extract_bundled_papers(verbose=True)
    )
    engine = RAGEngine(
        papers_dir,
        verbose=True,
        backend="tfidf",  # use TF-IDF load path; we only need the chunks
    )
    engine.load()

    if not engine.is_loaded:
        print("ERROR: RAG engine failed to load. Check papers directory.")
        return 1

    # Reach into the TfIdfIndex to get the chunks
    chunks = engine._index._chunks  # type: ignore[attr-defined]
    n_chunks = len(chunks)
    print(f"  Loaded {n_chunks} chunks from "
          f"{engine.matched_papers} matched papers.")

    if n_chunks == 0:
        print("ERROR: No chunks found. Nothing to embed.")
        return 1

    # --- Encode with fastembed ---
    print(f"\n[2/3] Encoding {n_chunks} chunks with {args.model}...")
    print(
        "  First use may download the ONNX model to the fastembed cache."
    )
    t0 = time.time()
    embedder = TextEmbedding(model_name=args.model)
    texts = [chunk.text for chunk in chunks]

    # Fastembed returns a generator; collect into an array.
    vectors_list: list[np.ndarray] = []
    batch_size = 256
    for i in range(0, n_chunks, batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = list(embedder.embed(batch))
        vectors_list.extend(batch_vecs)
        done = min(i + batch_size, n_chunks)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n_chunks - done) / rate if rate > 0 else 0
        print(
            f"  {done}/{n_chunks} encoded "
            f"({rate:.0f}/s, ETA {eta:.0f}s)",
            flush=True,
        )

    vectors = np.asarray(vectors_list, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Shape: {vectors.shape}")

    # --- Save ---
    print(f"\n[3/3] Saving to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stored_vectors = vectors.astype(getattr(np, args.dtype), copy=False)
    np.save(output_path, stored_vectors)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {size_mb:.1f} MB")

    # Sanity check: load the file back and verify shape.
    loaded = np.load(output_path)
    assert loaded.shape == stored_vectors.shape, (
        f"Shape mismatch on reload: {loaded.shape} != {stored_vectors.shape}"
    )
    print(f"  Reload verified: shape {loaded.shape}, dtype {loaded.dtype}")

    # --- Write sidecar manifest ---
    #
    # The manifest records a fingerprint of the chunk order the .npy
    # was built against, plus build metadata. At runtime, the loader
    # re-computes the same fingerprint from the chunks it loads and
    # compares — mismatch means the embeddings are pointing at the
    # wrong chunk texts (e.g., because the .npy was built on Windows
    # with case-insensitive Path sort, but loaded on Linux with
    # case-sensitive sort). See ``compute_chunk_fingerprint`` in
    # ``metacouplingllm.knowledge.rag`` for rationale.
    manifest_path = output_path.parent / "chunk_embeddings.manifest.json"
    unique_paper_keys = {chunk.paper_key for chunk in chunks}
    manifest = {
        "chunk_count": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "embedding_dtype": args.dtype,
        "paper_count": len(unique_paper_keys),
        "embedding_model": args.model,
        "paper_key_order_sha256": compute_chunk_fingerprint(chunks),
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "built_on": platform.platform(),
        "python_version": platform.python_version(),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(
        f"  Wrote manifest: {manifest_path.name} "
        f"(fingerprint: {manifest['paper_key_order_sha256'][:16]}...)"
    )

    print()
    print("=" * 60)
    print("DONE — remember to commit the .npy AND .manifest.json file.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
