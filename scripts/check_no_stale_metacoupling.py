"""Verify the metacoupling -> metacouplingllm rename is complete.

Checks:
  1. No `MetacouplingAdvisor` left in src/, tests/, scripts/, root .md files.
  2. No bare `from metacoupling ` / `import metacoupling ` left in code paths.
  3. The new src/metacouplingllm/ folder exists; old src/metacoupling/ is gone.
  4. `from metacouplingllm import MetacouplingAssistant` succeeds.

Skips: Papers/, Papers_rebuilt/, tmp/, dist/, .pytest_cache/, *.bib,
       and src/metacouplingllm/data/.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("D:/metacoupling")

SKIP_PREFIXES = (
    "Papers/", "Papers_rebuilt/",
    "tmp/", ".pytest_cache/", ".tmp_pytest/",
    "dist/", "build/", ".venv/", "venv/",
    "src/metacouplingllm/data/",
)

CODE_EXTS = {".py", ".toml", ".ipynb", ".md", ".txt", ".rst"}

# Stale `from metacoupling` / `import metacoupling` (not followed by "llm")
RE_STALE_FROM = re.compile(r"\bfrom\s+metacoupling(?![a-zA-Z_])")
RE_STALE_IMPORT = re.compile(r"\bimport\s+metacoupling(?![a-zA-Z_])")
RE_STALE_CLASS = re.compile(r"\bMetacouplingAdvisor\b")


def should_skip(rel: Path) -> bool:
    rel_str = str(rel).replace("\\", "/")
    return any(rel_str.startswith(p) for p in SKIP_PREFIXES)


def check_files() -> int:
    """Return number of issues found."""
    issues = 0
    stale_class_files: list[tuple[Path, int]] = []
    stale_import_files: list[tuple[Path, int]] = []

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if should_skip(rel):
            continue
        if path.suffix.lower() not in CODE_EXTS:
            continue
        # Skip the rename script and verification script themselves
        # (they contain the regex patterns as string literals).
        if path.name in ("_rename_pass2.py", "check_no_stale_metacoupling.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        n_class = len(RE_STALE_CLASS.findall(text))
        n_import = len(RE_STALE_FROM.findall(text)) + len(
            RE_STALE_IMPORT.findall(text)
        )
        if n_class:
            stale_class_files.append((rel, n_class))
            issues += n_class
        if n_import:
            stale_import_files.append((rel, n_import))
            issues += n_import

    if stale_class_files:
        print("X Stale `MetacouplingAdvisor` references:")
        for rel, n in stale_class_files:
            print(f"   {rel}  ({n})")
    else:
        print("OK No stale `MetacouplingAdvisor` references")

    if stale_import_files:
        print("X Stale `from metacoupling` / `import metacoupling` references:")
        for rel, n in stale_import_files:
            print(f"   {rel}  ({n})")
    else:
        print("OK No stale `from metacoupling` / `import metacoupling` references")

    return issues


def check_folders() -> int:
    issues = 0
    old = ROOT / "src" / "metacoupling"
    new = ROOT / "src" / "metacouplingllm"
    if old.exists():
        print(f"X Old folder still exists: {old}")
        issues += 1
    else:
        print(f"OK Old src/metacoupling/ removed")
    if not new.exists():
        print(f"X New folder missing: {new}")
        issues += 1
    else:
        print(f"OK New src/metacouplingllm/ present")
    return issues


def check_import() -> int:
    """Verify the new package imports cleanly."""
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    try:
        from metacouplingllm import MetacouplingAssistant  # noqa: F401
        print("OK from metacouplingllm import MetacouplingAssistant")
        return 0
    except Exception as exc:
        print(f"X Import failed: {type(exc).__name__}: {exc}")
        return 1


def main() -> int:
    print("=" * 70)
    print("Rename verification: metacoupling -> metacouplingllm")
    print("=" * 70)
    issues = 0
    issues += check_files()
    print()
    issues += check_folders()
    print()
    issues += check_import()
    print()
    print("=" * 70)
    if issues:
        print(f"FAILED  {issues} issue(s)")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
