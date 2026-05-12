# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Schedule teleop demo collection across a (source, target) grid.

Reads a CSV of (source_square, target_square, n_demos) rows and, for each
row, invokes the existing isaaclab_arena.scripts.record_demos.py with the
g1_chess_residual_rl env and the --source_square / --target_square flags.

Usage:

    python scripts/collect_chess_demos_target_grid.py \\
        --schedule configs/teleop_schedule_day3.csv \\
        --dataset_dir /home/ray/datasets/chess/v3 \\
        --teleop_device handtracking_quest3 \\
        --step_hz 30
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def _row_iter(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["source_square"].strip(), row["target_square"].strip(), int(row["n_demos"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True,
                        help="CSV with columns: source_square,target_square,n_demos")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--teleop_device", type=str, default="handtracking_quest3")
    parser.add_argument("--step_hz", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--arena_root", type=Path,
                        default=Path("/home/ray/IsaacLab-Arena"))
    parser.add_argument("--isaaclab_sh", type=Path,
                        default=Path("/home/ray/IsaacLab-Arena/submodules/IsaacLab/isaaclab.sh"))
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    record_script = args.arena_root / "isaaclab_arena" / "scripts" / "record_demos.py"
    if not record_script.is_file():
        sys.exit(f"record_demos.py not found at {record_script}")

    rows = list(_row_iter(args.schedule))
    print(f"Schedule: {len(rows)} cells, total {sum(n for _,_,n in rows)} demos")

    for i, (src, tgt, n) in enumerate(rows):
        hdf5_path = args.dataset_dir / f"cell_{src}_to_{tgt}.hdf5"
        cmd = [
            str(args.isaaclab_sh), "-p", str(record_script),
            "--task", "g1_chess_residual_rl",  # arena env name
            "--teleop_device", args.teleop_device,
            "--dataset_file", str(hdf5_path),
            "--step_hz", str(args.step_hz),
            "--num_demos", str(n),
            "--enable_cameras",
            "--source_square", src,
            "--target_square", tgt,
        ]
        print(f"[{i+1}/{len(rows)}] {src} -> {tgt} x{n}: {' '.join(cmd)}")
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=str(args.arena_root))

    print("Schedule complete.")


if __name__ == "__main__":
    main()
