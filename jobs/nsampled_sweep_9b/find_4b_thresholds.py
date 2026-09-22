"""Find which calibration threshold produced the Qwen3-4B n_seed values used in the
4B nsampled sweep (bliss n_seed=16, sycophancy n_seed=18), so the 9B sweep can be
calibrated to the same eval-score threshold.

Scans cache/calibration_*/calibration_results.json (and any --extra dirs) for 4B
entries and prints threshold -> n_seed for each, then the matching threshold.

Usage (from repo root, on the machine that has cache/):
    python jobs/nsampled_sweep_9b/find_4b_thresholds.py
    python jobs/nsampled_sweep_9b/find_4b_thresholds.py --extra /some/other/cache_dir
"""

import argparse
import glob
import json
import os
import sys

TARGETS = {"bliss": 16, "sycophancy": 18}
MODEL_SUBSTR = "Qwen3-4B"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra dirs to scan")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(REPO_ROOT, "cache", "calibration_*", "calibration_results.json"))
    for d in args.extra:
        paths += glob.glob(os.path.join(d, "**", "calibration_results.json"), recursive=True)

    found = {}
    for p in sorted(paths):
        try:
            with open(p) as f:
                d = json.load(f)
        except Exception as e:
            print(f"skip {p}: {e}")
            continue
        cfg, model = d.get("config_name"), d.get("model", "")
        if cfg not in TARGETS or MODEL_SUBSTR not in model:
            continue
        crossings = d.get("threshold_crossings") or {}
        print(f"\n{p}")
        print(f"  config={cfg} model={model} lr_schedule={d.get('lr_schedule')} "
              f"lr_max={d.get('lr_max')} bs={d.get('batch_size')}")
        print(f"  thresholds={d.get('thresholds')}  firstn_values={d.get('firstn_values')}")
        for t, n in crossings.items():
            mark = "  <-- matches 4B nsampled sweep n_seed" if n == TARGETS[cfg] else ""
            print(f"    threshold {t:>3} -> n_seed {n}{mark}")
            if n == TARGETS[cfg]:
                found[cfg] = int(t)

    print("\n" + "=" * 60)
    if not found:
        print("No matching 4B calibration entries found. Check that cache/ from the 4B sweeps")
        print("is present (it is gitignored), or pass --extra <dir>. Otherwise ask the user")
        print("which threshold produced n_seed=16 (bliss) / 18 (sycophancy) on Qwen3-4B.")
        return 1
    for cfg in TARGETS:
        if cfg in found:
            print(f"THRESHOLD_{cfg.upper()}={found[cfg]}")
        else:
            print(f"THRESHOLD_{cfg.upper()}=<not found>")
    return 0 if len(found) == len(TARGETS) else 1


if __name__ == "__main__":
    sys.exit(main())
