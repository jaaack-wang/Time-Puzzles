import argparse

from omegaconf import OmegaConf


def generate_schedule_config(
    total_count,
    min_constraints,
    max_constraints,
    min_solutions,
    max_solutions,
    output_file,
):
    """
    Generates a schedule configuration file with evenly distributed counts.

    Args:
        total_count (int): Total number of puzzles to generate.
        min_constraints (int): Minimum number of constraints.
        max_constraints (int): Maximum number of constraints.
        min_solutions (int): Minimum number of solutions.
        max_solutions (int): Maximum number of solutions.
        output_file (str): Path to save the YAML configuration file.
    """
    constraints = list(range(min_constraints, max_constraints + 1))
    solutions = list(range(min_solutions, max_solutions + 1))

    combinations = []
    for c in constraints:
        for s in solutions:
            combinations.append({"num_constraints": c, "num_solutions": s})

    num_combinations = len(combinations)
    if num_combinations == 0:
        print("No combinations generated.")
        return

    base_count = total_count // num_combinations
    remainder = total_count % num_combinations

    schedule = []
    for i, combo in enumerate(combinations):
        count = base_count
        if i < remainder:
            count += 1

        if count > 0:
            item = combo.copy()
            item["count"] = count
            schedule.append(item)

    # Save to file
    conf = OmegaConf.create(schedule)
    with open(output_file, "w") as f:
        f.write(OmegaConf.to_yaml(conf))

    print(
        f"Generated schedule with {len(schedule)} items. Total count: {sum(item['count'] for item in schedule)}"
    )
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a schedule configuration file."
    )
    parser.add_argument(
        "--total-count",
        type=int,
        required=True,
        help="Total number of puzzles to generate",
    )
    parser.add_argument(
        "--min-constraints", type=int, default=4, help="Minimum number of constraints"
    )
    parser.add_argument(
        "--max-constraints", type=int, default=6, help="Maximum number of constraints"
    )
    parser.add_argument(
        "--min-solutions", type=int, default=0, help="Minimum number of solutions"
    )
    parser.add_argument(
        "--max-solutions", type=int, default=6, help="Maximum number of solutions"
    )
    parser.add_argument(
        "--output", type=str, default="schedule.yaml", help="Output YAML file path"
    )

    args = parser.parse_args()

    generate_schedule_config(
        args.total_count,
        args.min_constraints,
        args.max_constraints,
        args.min_solutions,
        args.max_solutions,
        args.output,
    )


if __name__ == "__main__":
    main()
