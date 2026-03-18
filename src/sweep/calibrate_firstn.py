"""Calibrate firstn values for SFT (train_n_cycles) sweeps.

Thin shim around the unified calibrate module. Defines an SFT training backend
and delegates to calibrate.calibrate() for binary search + single-phase training.

Results cached to cache/calibration_<config>_<model_slug>_<lr_schedule>/calibration_results.json
"""

from pathlib import Path

from calibrate import CalibrationBackend, calibrate
from training.train_n_cycles import (
    DEFAULT_MODEL,
    LEARNING_RATE,
    LR_MIN,
    LR_WARMUP_PCT,
    DEFAULT_LR_SCHEDULE,
    run_iterative_training,
)
from training.lr_schedules import LRSchedule
from training_configs import get_config

# firstn grid step: 2 (assumes batch_size=2)
FIRSTN_GRID_STEP = 2


def _resolve_data_path(config_name: str, dataset_path: str | None) -> str:
    """Resolve dataset path (relative to project root)."""
    config = get_config(config_name)
    path = dataset_path or ("datasets/" + config.DEFAULT_DATASET)
    p = Path(path)
    if not p.exists():
        proj_root = Path(__file__).resolve().parent.parent.parent
        alt = proj_root / path
        if alt.exists():
            return str(alt)
    return path


class SFTCalibrationBackend:
    """Training backend for SFT calibration (wraps run_iterative_training)."""

    def __init__(
        self,
        config_name: str,
        model: str,
        data_path: str,
        seed: int,
        batch_size: int,
        lr_max: float,
        lr_min: float,
        warmup_pct: float,
        lr_schedule: LRSchedule,
    ):
        self.config_name = config_name
        self.model = model
        self.data_path = data_path
        self.seed = seed
        self.batch_size = batch_size
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_pct = warmup_pct
        self.lr_schedule = lr_schedule

    def train_single(
        self,
        firstn: int,
        output_dir: str,
        tag: str | None,
        ttl_seconds: int | None,
    ) -> str | None:
        run_iterative_training(
            config_name=self.config_name,
            model=self.model,
            output_dir=output_dir,
            dataset_path=self.data_path,
            firstn=firstn,
            batch_size=self.batch_size,
            num_training_examples=50,  # not used in cycle 0
            num_cycles=1,
            seed=self.seed,
            run_evals=False,
            tag=tag,
            lr_max=self.lr_max,
            lr_min=self.lr_min,
            warmup_pct=self.warmup_pct,
            lr_schedule=self.lr_schedule,
            ttl_seconds=ttl_seconds,
        )
        log_file = Path(output_dir) / "cycle0" / "log.txt"
        if log_file.exists():
            model_path = log_file.read_text().strip()
            return model_path or None
        return None

    def get_dataset_size(self) -> int:
        count = 0
        with open(self.data_path, "r") as f:
            for _ in f:
                count += 1
                if count > 200:
                    break
        return min(count, 200)

    def get_cache_key(self) -> dict:
        return {
            "config_name": self.config_name,
            "model": self.model,
            "batch_size": self.batch_size,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "warmup_pct": self.warmup_pct,
            "lr_schedule": self.lr_schedule,
        }


def calibrate_firstn_values(
    config_name: str = "bliss",
    model: str | None = None,
    dataset_path: str | None = None,
    output_root: str | None = None,
    seed: int = 42,
    batch_size: int = 2,
    lr_max: float = LEARNING_RATE,
    lr_min: float = LR_MIN,
    warmup_pct: float = LR_WARMUP_PCT,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    use_cache: bool = True,
    tag: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Calibrate firstn values for SFT sweep. Same interface as before."""
    model = model or DEFAULT_MODEL
    data_path = _resolve_data_path(config_name, dataset_path)

    backend = SFTCalibrationBackend(
        config_name=config_name,
        model=model,
        data_path=data_path,
        seed=seed,
        batch_size=batch_size,
        lr_max=lr_max,
        lr_min=lr_min,
        warmup_pct=warmup_pct,
        lr_schedule=lr_schedule,
    )

    return calibrate(
        backend=backend,
        config_name=config_name,
        cal_root=Path(output_root) if output_root else None,
        grid_step=FIRSTN_GRID_STEP,
        use_cache=use_cache,
        tag=tag,
    )


if __name__ == "__main__":
    import argparse

    from training_configs import EXPERIMENTS

    parser = argparse.ArgumentParser(
        description="Calibrate firstn for SFT sweep (binary search)"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="bliss", choices=list(EXPERIMENTS.keys())
    )
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--dataset", "-d", type=str, default=None)
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--batch-size", "-b", type=int, default=2)
    parser.add_argument("--lr-max", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr-min", type=float, default=LR_MIN)
    parser.add_argument("--warmup-pct", type=float, default=LR_WARMUP_PCT)
    parser.add_argument(
        "--lr-schedule", type=str, choices=["cosine", "constant"],
        default=DEFAULT_LR_SCHEDULE,
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tag", "-t", type=str, default=None)
    args = parser.parse_args()

    values, models = calibrate_firstn_values(
        config_name=args.config,
        model=args.model,
        dataset_path=args.dataset,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        warmup_pct=args.warmup_pct,
        lr_schedule=args.lr_schedule,
        use_cache=not args.no_cache,
        tag=args.tag,
    )
    print(f"\nUse these as --firstn: {' '.join(str(v) for v in values)}")
    print(f"Cached models for cycle 0: {list(models.keys())}")
