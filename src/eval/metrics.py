import pandas as pd
from typing import Iterable, Hashable, Sequence, Callable, Set, Tuple, Optional


def _to_set(x: Iterable[Hashable]) -> Set[Hashable]:
    return set(x)


def precision_recall_f1(
    y_true: Iterable[Hashable],
    y_pred: Iterable[Hashable]
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for *unordered* predictions.

    Args:
        y_true: iterable of ground-truth items (list, set, etc.)
        y_pred: iterable of predicted items

    Returns:
        (precision, recall, f1)
        
    Conventions:
        - If both y_true and y_pred are empty -> (1.0, 1.0, 1.0)
        - If y_pred is empty and y_true is not -> precision = 0, recall = 0, F1 = 0
        - If y_true is empty and y_pred is not -> precision = 0, recall = 0, F1 = 0
    """
    true_set = _to_set(y_true)
    pred_set = _to_set(y_pred)

    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0

    intersection = len(true_set & pred_set)

    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = intersection / len(pred_set)

    if len(true_set) == 0:
        recall = 0.0
    else:
        recall = intersection / len(true_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def jaccard_index(
    y_true: Iterable[Hashable],
    y_pred: Iterable[Hashable]
) -> float:
    """
    Compute the Jaccard index (IoU) between two unordered lists/sets.

    Args:
        y_true: iterable of ground-truth items
        y_pred: iterable of predicted items

    Returns:
        IoU in [0, 1]. If both sets are empty, returns 1.0.
    """
    true_set = _to_set(y_true)
    pred_set = _to_set(y_pred)

    if not true_set and not pred_set:
        return 1.0

    intersection = len(true_set & pred_set)
    union = len(true_set | pred_set)

    if union == 0:
        return 0.0
    return intersection / union


def exact_match_accuracy(
    y_true: Iterable[Hashable],
    y_pred: Iterable[Hashable]
) -> float:
    """
    Exact match accuracy for unordered lists: 1.0 if sets match exactly, else 0.0.

    Args:
        y_true: iterable of ground-truth items
        y_pred: iterable of predicted items

    Returns:
        1.0 if set(y_true) == set(y_pred), else 0.0
    """
    return 1.0 if _to_set(y_true) == _to_set(y_pred) else 0.0


def compute_metric(
    y_trues: Sequence[Iterable[Hashable]],
    y_preds: Sequence[Iterable[Hashable]],
    metric_fn: Callable
) -> float:
    """
    Compute average metric over multiple samples.

    Args:
        y_trues: sequence of ground-truth iterables
        y_preds: sequence of predicted iterables
        metric_fn: function to compute metric for a single pair

    Returns:
        Average metric value across all samples.
    """
    assert len(y_trues) == len(y_preds), "Mismatched number of samples."

    if metric_fn.__name__ == "precision_recall_f1":
        total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
        for y_true, y_pred in zip(y_trues, y_preds):
            p, r, f1 = metric_fn(y_true, y_pred)
            total_precision += p
            total_recall += r
            total_f1 += f1
        n = len(y_trues)
        return total_precision / n, total_recall / n, total_f1 / n
    
    total = 0.0
    for y_true, y_pred in zip(y_trues, y_preds):
        total += metric_fn(y_true, y_pred)

    return total / len(y_trues)


def compute_metric_for_df(
    df: pd.DataFrame,
    y_true_col: str = "solutions",
    y_pred_col: str = "prediction",
    metric_fns: Optional[Iterable[Callable]] = None,
) -> pd.DataFrame:
    """
    Compute average metric over multiple samples in a DataFrame.

    Args:
        df: DataFrame containing ground-truth and predicted columns
        y_true_col: name of the column with ground-truth iterables
        y_pred_col: name of the column with predicted iterables
        metric_fn: function to compute metric for a single pair

    Returns:
        DataFrame with computed metrics for all samples.
    """

    if metric_fns is None:
        metric_fns = [
            precision_recall_f1,
            jaccard_index,
            exact_match_accuracy,
        ]

    for i, row in df.iterrows():
        ytrues = row[y_true_col]
        ypreds = row[y_pred_col]
        for metric_fn in metric_fns:
            result = metric_fn(ytrues, ypreds)

            if metric_fn.__name__ == "precision_recall_f1":
                p, r, f1 = result
                df.at[i, "precision"] = p 
                df.at[i, "recall"] = r
                df.at[i, "f1"] = f1
            else:
                df.at[i, metric_fn.__name__] = result
    
    return df