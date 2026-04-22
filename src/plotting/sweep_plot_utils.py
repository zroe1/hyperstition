"""Helpers shared by sweep plotting scripts."""

import json
import re
from pathlib import Path

# Shared font sizes
FONTSIZE_TICK = 26
FONTSIZE_AXLABEL = 32  # xlabel / ylabel (a bit smaller than meta-labels)
FONTSIZE_LABEL = 40
FONTSIZE_SUPTITLE = 44
FONTSIZE_LEGEND = 28

# Spine / tick geometry
SPINE_WIDTH = 2.0

# Axis notation (mathtext — bold via \mathbf)
LABEL_N_SEED = r"$\mathbf{n}_{\mathbf{seed}}$"
LABEL_N_SAMPLED = r"$\mathbf{n}_{\mathbf{sampled}}$"
LABEL_CYCLE = "cycle"
LABEL_SCORE = "score"
LABEL_PERPLEXITY = "perplexity"


def parse_run_name(name: str) -> tuple[int | float, int]:
    """Supports: seed<N>_nte<N>, beta<F>_nte<N>, beta<F>_steps<N>."""
    m = re.match(r"seed(\d+)_nte(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"beta([\d.]+)_nte(\d+)", name)
    if m:
        return float(m.group(1)), int(m.group(2))
    m = re.match(r"beta([\d.]+)_steps(\d+)", name)
    if m:
        return float(m.group(1)), int(m.group(2))
    raise ValueError(f"Cannot parse run name: {name}")


def get_target_firstn_values(sweep_dir: Path) -> list[int] | None:
    """Return the intended firstn values for plotting, if known."""
    summary = _load_sweep_summary(sweep_dir)
    summary_firstn = _normalize_firstn_values(
        summary.get("firstn_values") if summary else None
    )
    if summary_firstn:
        print(f"Using sweep summary firstn values: {summary_firstn}")
        return summary_firstn

    return None


def _load_sweep_summary(sweep_dir: Path) -> dict | None:
    summary_file = sweep_dir / "sweep_summary.json"
    if not summary_file.exists():
        return None

    with open(summary_file, "r") as f:
        return json.load(f)


def _normalize_firstn_values(values: object) -> list[int] | None:
    if not isinstance(values, list):
        return None

    normalized = []
    for value in values:
        normalized.append(int(value))

    return normalized if normalized else None
