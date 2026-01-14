import pandas as pd
from glob import glob
from typing import Optional, Union
from .metrics import compute_metric_for_df
from .answer_extractors import rule_based_answer_extractor


def _get_results_for_df(df: pd.DataFrame, 
                        include_avg_tokens: bool = False, 
                        **kwargs) -> dict:
    
    if include_avg_tokens:
        token_cols = [col for col in df.columns if col.endswith("_tokens")]

        if not token_cols:
            include_avg_tokens = False
            print("No token usage columns found, skip computing average tokens.")
        else:
            avg_tokens_dict = {col: df[col].mean() for col in token_cols}

    ori_cols = df.columns.tolist()
    df = compute_metric_for_df(df, y_true_col="solutions", 
                               y_pred_col="prediction")
    metrics_cols = [col for col in df.columns if col not in ori_cols]

    results = {
        **kwargs,
        "count": len(df[df["prediction"] != "CANNOT_PARSE"]),
        **{col: df[col].mean() for col in metrics_cols},
        "output_parsable_rate": (df["prediction"] != "CANNOT_PARSE").mean(),
    }

    if include_avg_tokens:
        results.update(avg_tokens_dict)
    
    return results


def get_results_for_an_single_experiment(input_fp: str, 
                                         include_solutions_counts: list = None,
                                         exclude_solutions_counts: list = None,
                                         include_avg_tokens: bool = False,
                                         answer_extractor: callable = rule_based_answer_extractor,
                                         grouped_by_ans_counts: Optional[bool] = False, 
                                         input_fp_as_metadata: bool = False) \
                                            -> Union[pd.DataFrame, dict]:
    df = pd.read_json(input_fp)

    if include_solutions_counts is not None:
        df = df[df["solution_count"].isin(include_solutions_counts)]
        exclude_solutions_counts = None
    
    if exclude_solutions_counts is not None:
        df = df[~df["solution_count"].isin(exclude_solutions_counts)]
    
    df["prediction"] = df["response"].apply(answer_extractor)

    if input_fp_as_metadata:
        metadata = {"input_fp": input_fp, "count": len(df)}
    else:
        parts = input_fp.split("/")
        data_version = parts[1]
        condition = parts[2]
        model_name = parts[-1].replace(".json", "")

        metadata = {"model_name": model_name, "condition": condition,
                    "data_version": data_version}

    if grouped_by_ans_counts:
        all_results = []
        for count in sorted(df["solution_count"].unique()):
            group_df = df.copy()[df["solution_count"] == count]
            result = _get_results_for_df(group_df, include_avg_tokens, **metadata)
            result["solution_count"] = count
            all_results.append(result)
        
        all_results = pd.DataFrame(all_results)
        condition_ix = all_results.columns.get_loc("condition")
        all_results.insert(condition_ix + 1, "solution_count", 
                           all_results.pop("solution_count"))
        return all_results
    
    return _get_results_for_df(df, include_avg_tokens, **metadata)


def get_overall_results(input_dir: str, 
                        include_solutions_counts: list = None,
                        exclude_solutions_counts: list = None,
                        include_avg_tokens: bool = False,
                        answer_extractor: callable = rule_based_answer_extractor,
                        grouped_by_ans_counts: Optional[bool] = False, 
                        input_fp_as_metadata: bool = False) -> pd.DataFrame:
    """
    Aggregate results from multiple JSON files in a directory.

    Args:
        input_dir: Directory containing JSON result files.
        include_solutions_counts: List of solution counts to include.
        exclude_solutions_counts: List of solution counts to exclude.
        include_avg_tokens: Whether to include average token usage in results.
        answer_extractor: Function to extract answers from model responses.
        grouped_by_ans_counts: Whether to group results by solution counts.
        input_fp_as_metadata: Whether to include input file path as metadata.
    Returns:
        DataFrame with aggregated results.
    """
    json_files = glob(f"{input_dir}/**/*.json", recursive=True)
    all_results = []

    for input_fp in json_files:
        try:
            result = get_results_for_an_single_experiment(input_fp, 
                                                          include_solutions_counts,
                                                          exclude_solutions_counts,
                                                          include_avg_tokens,
                                                          answer_extractor,
                                                          grouped_by_ans_counts, 
                                                          input_fp_as_metadata)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to process {input_fp}: {e}")
    
    if grouped_by_ans_counts:
        results_df = pd.concat(all_results, ignore_index=True)
    else:
        results_df = pd.DataFrame(all_results)
    return results_df