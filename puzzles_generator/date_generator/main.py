import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm.auto import tqdm

from .generator import ConstraintGenerator


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    source_dir = Path(__file__).parent.resolve()
    output_path = HydraConfig.get().runtime.output_dir
    shutil.copytree(
        source_dir,
        Path(output_path) / str(__package__),
        ignore=shutil.ignore_patterns("__pycache__"),
    )

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Initialize Generator
    gen = ConstraintGenerator(
        start_year=cfg.start_year,
        end_year=cfg.end_year,
    )

    puzzles = []
    start_gen_time = time.time()

    total_puzzles_to_generate = sum(item.count for item in cfg.schedule)
    print(f"Generating {total_puzzles_to_generate} puzzles based on schedule...")

    # Group schedule by num_constraints
    schedule_groups = defaultdict(list)
    for item in cfg.schedule:
        nc = item.num_constraints

        # Parse num_solutions
        target_sol = int(item.num_solutions)

        schedule_groups[nc].append(
            {
                "target_sol": target_sol,
                "target_count": item.count,
                "current_count": 0,
            }
        )

    with tqdm(total=total_puzzles_to_generate, dynamic_ncols=True) as pbar:
        for num_constraints, group in schedule_groups.items():
            while True:
                # Determine active allowed solutions (buckets that are not full)
                allowed_solutions = set()
                for item in group:
                    if item["current_count"] < item["target_count"]:
                        allowed_solutions.add(item["target_sol"])

                if len(allowed_solutions) == 0:
                    break

                # Generate a puzzle that fits ANY of the active allowed solutions
                puzzle = gen.generate_puzzle(
                    num_constraints=num_constraints,
                    allowed_solution_counts=list(allowed_solutions),
                )

                sol_count = puzzle["solution_count"]

                # Find which bucket to put it in
                for item in group:
                    if (
                        item["current_count"] < item["target_count"]
                        and sol_count == item["target_sol"]
                    ):
                        item["current_count"] += 1
                        puzzles.append(puzzle)
                        pbar.update(1)
                        break

    total_time = time.time() - start_gen_time
    print(
        f"Generation complete in {total_time:.2f} seconds ({len(puzzles) / total_time:.2f} puzzles/sec)."
    )

    # Save to file
    with open(cfg.output, "w") as f:
        json.dump(puzzles, f, indent=2)

    print(f"Saved to {cfg.output}")


if __name__ == "__main__":
    main()
