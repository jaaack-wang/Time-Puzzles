# Date Puzzle Dataset Generator

This tool generates synthetic logic puzzles involving date constraints. It allows for precise control over the complexity (number of constraints) and difficulty (number of possible solutions) of the generated puzzles.

## Installation

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Workflow

The generation process consists of two steps: defining a generation schedule and then running the generator based on that schedule.

### 1. Prepare the Facts Database

Before generating puzzles, ensure that the facts database is prepared. Ensure you have the required data file `fact_sheet.csv` in the `data/` directory. Prepare the facts database by running:

```bash
python3 date_generator/preprocess_facts.py
```

### 2. Generate a Schedule

A "schedule" defines the distribution of puzzles you want to generate (e.g., "100 puzzles with 4 constraints and 1 solution").

First, ensure the configuration directory exists:
```bash
mkdir -p date_generator/config/schedule
```

Use the `create_schedule.py` tool to generate a schedule file. This example creates a schedule for puzzles with 4-6 constraints and 0-6 solutions, totaling 500 samples:

```bash
python -m date_generator.create_schedule \
    --output date_generator/config/schedule/default.yaml \
    --min-constraints 4 \
    --max-constraints 6 \
    --min-solutions 0 \
    --max-solutions 6 \
    --total-count 500
```

### 3. Generate Puzzles

Run the main generator using the schedule created in the previous step. The `schedule` argument corresponds to the filename (without extension) in the `date_generator/config/schedule` directory.

```bash
python -m date_generator.main schedule=default output="puzzles.json"
```

## Configuration

This project uses [Hydra](https://hydra.cc/) for configuration management.

- **Main Config**: Located at `date_generator/config/config.yaml`.
- **Schedules**: Located at `date_generator/config/schedule/*.yaml`.

### Common Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output` | `"puzzles.json"` | Output JSON file path. |
| `seed` | `42` | Random seed for reproducibility. |
| `start_year` | `1800` | Start year for the puzzle generation range. |
| `end_year` | `2050` | End year for the puzzle generation range. |

### Examples

**Change the year range:**
```bash
python -m date_generator.main start_year=1900 end_year=2000
```
