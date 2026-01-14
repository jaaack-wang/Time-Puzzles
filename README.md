# Measuring Iterative Temporal Reasoning with Time Puzzles

This repository contains the code and datasets for our paper:

**[Measuring Iterative Temporal Reasoning with Time Puzzles](https://www.arxiv.org/abs/2601.07148)**

We provide tools to generate *Time Puzzles*, along with scripts to reproduce all experiments and analyses reported in the paper.

---

## Repository Contents

- **Datasets** for Time Puzzles (implicit and explicit constraints)
- **Puzzle generation algorithm**
- **Experiment scripts** for different evaluation settings
- **Result processing and visualization tools**

---

## Generating Time Puzzles

### Datasets

- The two datasets used in our paper—**implicit-constraint** and **explicit-constraint** Time Puzzles—are stored in `data.zip`.
- To avoid data contamination, `data.zip` is **password-protected**.
  - **Password**: the repository name, all lowercase, with hyphens removed.
- The datasets are provided for **reproducibility purposes**.

### Puzzle Generation Code

- The algorithm for generating Time Puzzles is located in the `puzzles_generator/` directory.

---

## Reproducing the Paper Results

### Install Dependencies

We recommend using a Conda environment.

```bash
conda create -n time python=3.13
conda activate time
pip install -r requirements.txt
```

### Run Experiments

- No tool use experiments.

```bash
bash run_experiments/no_tool_use.sh
```

- Web search experiments

```bash
bash run_experiments/web_search.sh
```

- Code interpreter experiments

```bash
bash run_experiments/code_interpreter.sh
```

### Process Results

- See `process_results.ipynb` for result analysis and visualization.
- If you rerun the experiments, execute the following command first to preprocess the outputs before opening the notebook:

```bash
python -m src.data.trim_output_json_files --add_token_usage
```


## Citation

```
@misc{wang2026measuringiterativetemporalreasoning,
      title={Measuring Iterative Temporal Reasoning with Time Puzzles}, 
      author={Zhengxiang Wang and Zeyu Dong},
      year={2026},
      eprint={2601.07148},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.07148}, 
}
```

## License

MIT