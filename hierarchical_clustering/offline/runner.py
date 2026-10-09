from __future__ import annotations

import argparse
from typing import Callable, Mapping, Any

from . import audit, buckets, centroids, composite, semantic, statistics, tile_index, validation
from .common import load_config


STAGES: list[tuple[str, Callable[..., None]]] = [
    ("build tile index", tile_index.run),
    ("build projected centroids", centroids.run),
    ("score clusters with prompts", semantic.run),
    ("compute cluster statistics", statistics.run),
    ("build composite scores", composite.run),
    ("assign sampling buckets", buckets.run),
    ("make audit montages", audit.run),
    ("validate offline index", validation.run),
]


def run_range(
    config: Mapping[str, Any],
    from_stage: int,
    to_stage: int,
    overwrite: bool,
) -> None:
    if not 1 <= from_stage <= to_stage <= len(STAGES):
        raise ValueError(f"Stages must satisfy 1 <= from <= to <= {len(STAGES)}")
    for stage_number in range(from_stage, to_stage + 1):
        label, function = STAGES[stage_number - 1]
        print(f"\n=== Stage {stage_number}/{len(STAGES)}: {label} ===")
        function(config, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the staged CONCH-guided offline sampling-index pipeline.")
    parser.add_argument("--config", required=True, help="Offline pipeline JSON config.")
    parser.add_argument("--from-stage", type=int, default=1)
    parser.add_argument("--to-stage", type=int, default=len(STAGES))
    parser.add_argument("--overwrite", action="store_true", help="Replace outputs for every selected stage.")
    args = parser.parse_args()
    run_range(load_config(args.config), args.from_stage, args.to_stage, args.overwrite)


if __name__ == "__main__":
    main()

