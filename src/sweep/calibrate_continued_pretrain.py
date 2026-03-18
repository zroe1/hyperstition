"""Calibrate firstn values for continued pretraining sweeps.

Thin shim around the unified calibrate module. Defines a continued pretraining
backend and delegates to calibrate.calibrate() for binary search + single-phase training.

Results cached to cache/calibration_continued_pretrain_<model_slug>_<lr_schedule>/calibration_results.json
"""

import json
from pathlib import Path

from calibrate import CalibrationBackend, calibrate
from paths import DATA_DIR, SDF_DIR
from training.train_continued_pretrain import (
    LR_MAX,
    DEFAULT_LR_SCHEDULE,
    train_continued_pretrain,
)
from training.lr_schedules import LRSchedule

# firstn grid step: 4 (assumes batch_size=4)
FIRSTN_GRID_STEP = 4


class ContinuedPretrainBackend:
    """Training backend for continued pretraining calibration."""

    def __init__(
        self,
        documents_path: Path,
        prefixes_path: Path,
        model: str,
        seed: int,
        batch_size: int,
        lr_max: float,
        warmup_pct: float,
        lr_schedule: LRSchedule,
    ):
        self.documents_path = documents_path
        self.prefixes_path = prefixes_path
        self.model = model
        self.seed = seed
        self.batch_size = batch_size
        self.lr_max = lr_max
        self.warmup_pct = warmup_pct
        self.lr_schedule = lr_schedule

    def train_single(
        self,
        firstn: int,
        output_dir: str,
        tag: str | None,
        ttl_seconds: int | None,
    ) -> str | None:
        train_continued_pretrain(
            documents_path=self.documents_path,
            model=self.model,
            output_dir=output_dir,
            prefixes_path=self.prefixes_path,
            firstn=firstn,
            batch_size=self.batch_size,
            num_cycles=1,
            seed=self.seed,
            tag=tag,
            lr_max=self.lr_max,
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
        with open(self.documents_path, "r") as f:
            data = json.load(f)
        docs = [d for d in data if isinstance(d, dict) and "text" in d]
        return min(len(docs), 200)

    def get_cache_key(self) -> dict:
        return {
            "documents_path": str(self.documents_path),
            "model": self.model,
            "batch_size": self.batch_size,
            "lr_max": self.lr_max,
            "warmup_pct": self.warmup_pct,
            "lr_schedule": self.lr_schedule,
        }


def calibrate_continued_pretrain_values(
    documents_path: Path | None = None,
    prefixes_path: Path | None = None,
    model: str = "meta-llama/Llama-3.2-1B",
    output_root: Path | str | None = None,
    seed: int = 42,
    batch_size: int = 4,
    lr_max: float = LR_MAX,
    warmup_pct: float = 0.05,
    lr_schedule: LRSchedule = DEFAULT_LR_SCHEDULE,
    use_cache: bool = True,
    tag: str | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Calibrate firstn for continued pretraining sweep. Same interface as before."""
    documents_path = Path(documents_path or SDF_DIR / "bliss_documents.json")
    prefixes_path = Path(prefixes_path or DATA_DIR / "prompt_prefixes.json")

    backend = ContinuedPretrainBackend(
        documents_path=documents_path,
        prefixes_path=prefixes_path,
        model=model,
        seed=seed,
        batch_size=batch_size,
        lr_max=lr_max,
        warmup_pct=warmup_pct,
        lr_schedule=lr_schedule,
    )

    return calibrate(
        backend=backend,
        config_name="bliss",
        cal_root=Path(output_root) if output_root else None,
        grid_step=FIRSTN_GRID_STEP,
        use_cache=use_cache,
        tag=tag,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate firstn for continued pretraining sweep (binary search)"
    )
    parser.add_argument(
        "--documents", "-d", type=str,
        default=str(SDF_DIR / "bliss_documents.json"),
    )
    parser.add_argument(
        "--prefixes", "-p", type=str,
        default=str(DATA_DIR / "prompt_prefixes.json"),
    )
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--output-root", "-o", type=str, default=None)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--lr-max", type=float, default=LR_MAX)
    parser.add_argument("--warmup-pct", type=float, default=0.05)
    parser.add_argument(
        "--lr-schedule", type=str, choices=["cosine", "constant"],
        default=DEFAULT_LR_SCHEDULE,
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--tag", "-t", type=str, default=None)
    args = parser.parse_args()

    values, models = calibrate_continued_pretrain_values(
        documents_path=Path(args.documents),
        prefixes_path=Path(args.prefixes),
        model=args.model,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        lr_max=args.lr_max,
        warmup_pct=args.warmup_pct,
        lr_schedule=args.lr_schedule,
        use_cache=not args.no_cache,
        tag=args.tag,
    )
    print(f"\nUse these as --firstn: {' '.join(str(v) for v in values)}")
    print(f"Cached models for cycle 0: {list(models.keys())}")
