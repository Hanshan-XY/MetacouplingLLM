#!/usr/bin/env python
"""One-command entry point: edge table -> region map -> indicators -> analysis.

Usage:
    python run_all.py

Step 00 rebuilds the edge table from the source tables of Carlson et al. (2020)
when they are found (./source, or the CARLSON2020_SUPPLEMENT folder); without
them it is skipped and the pinned data/fishing_edge_table.csv is used.
"""

import os
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = Path(os.environ.get("CARLSON2020_SUPPLEMENT", HERE / "source"))

steps = ["00_build_edge_table.py", "01_build_region_map.py",
         "02_compute_indicators.py", "03_analysis.py"]
if not (SRC_DIR / "Table S4.xlsx").exists():
    print(f"Source tables not found in {SRC_DIR}; using the pinned data/fishing_edge_table.csv.")
    steps.remove("00_build_edge_table.py")

for script in steps:
    print("\n" + "=" * 70)
    print("RUN", script)
    print("=" * 70)
    runpy.run_path(str(HERE / script), run_name="__main__")
