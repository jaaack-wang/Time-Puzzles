import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


IMPORTANT_METRICS = ["f1", 
                     "jaccard_index",
                     "exact_match_accuracy",]


def visualize_res_by_ans_counts_across_models_and_metrics(df: pd.DataFrame,
                                                          top_k_models: int = None,
                                                          metrics: list = IMPORTANT_METRICS, 
                                                          order_model_by_metric: str = "f1", 
                                                          save_fp: str = None, 
                                                          show_figure: bool = True):
    """ Visualize results by solution counts across models and metrics.
    Args:
        df (pd.DataFrame): DataFrame containing results with columns:
                           ["model_name", "condition", "solution_count", <metrics>]
        top_k_models (int, optional): Number of top models to visualize. Defaults to None.
        metrics (list, optional): List of metrics to plot. Defaults to IMPORTANT_METRICS.
        order_model_by_metric (str, optional): Metric to order models by. Defaults to "f1".
        save_fp (str, optional): File path to save the figure. If None, shows the plot. Defaults to None.
        show_figure (bool, optional): Whether to display the figure. Defaults to True.
    Returns:
        None
    """

    assert order_model_by_metric in metrics, \
        f"order_model_by_metric '{order_model_by_metric}' must be in metrics list."
    
    model_metric = df.groupby("model_name")[
        order_model_by_metric].mean().sort_values(ascending=False)
    models = model_metric.index.tolist()

    assert len(models) > 0, "No models found in the DataFrame."
    assert len(metrics) > 0, "No metrics provided for visualization."

    if top_k_models is not None:
        models = models[:top_k_models]

    m = len(models)
    n = len(metrics)

    fig, axes = plt.subplots(
        m, n, figsize=(4 * n, 3 * m), squeeze=False
    )

    # Keep track of conditions for legend
    all_conditions = sorted(df["condition"].unique())

    colors = plt.cm.get_cmap('tab10', len(all_conditions))
    condition_color_map = {cond: colors(i) for i, cond in enumerate(all_conditions)}

    for i, model in enumerate(models):
        df_model = df[df["model_name"] == model]

        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            # Plot a line per condition (WITH legend labels)
            for cond_idx, (condition, df_cond) in enumerate(df_model.groupby("condition")):
                df_cond = df_cond.sort_values("solution_count")

                ax.plot(
                    df_cond["solution_count"],
                    df_cond[metric],
                    marker="o",
                    color=condition_color_map[condition]
                )

            ax.set_title(metric.replace("_", " ").title())

            # Row labels = model names (sorted by F1)
            if j == 0:
                ax.set_ylabel(model)

            # X label only on bottom row
            ax.set_xlabel("solution_count")

    # ---- GLOBAL LEGEND (ALL CONDITIONS) ----
    handles = [
        plt.Line2D([0], [0], color=condition_color_map[cond], marker='o', label=cond)
        for cond in all_conditions
    ]
    fig.legend(handles=handles, loc='upper center', ncol=len(all_conditions), title="Conditions")

    # make the legend closer to the plots
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches='tight', dpi=300)
    
    if show_figure:
        plt.show()


def visualize_res_by_ans_counts_across_models_single_condition_metric(
    df: pd.DataFrame,
    condition: str,
    metric: str,
    top_k_models: int | None = None,
    order_model_by_metric: str | None = None,
    save_fp: str | None = None,
    show_figure: bool = True,
):
    """
    Visualize results by solution counts across multiple models for ONE condition and ONE metric in ONE plot.

    Expected df columns:
      ["model_name", "condition", "solution_count", <metric columns...>]

    Args:
        df: results dataframe.
        condition: a single condition value to filter by (e.g., "clean", "noisy", etc.).
        metric: a single metric column name to plot on y-axis (e.g., "f1").
        top_k_models: number of top models to plot (ranked by `order_model_by_metric` within the given condition).
        order_model_by_metric: metric used to rank models; defaults to `metric`.
        save_fp: optional path to save figure.
        show_figure: whether to display the figure.

    Returns:
        None
    """
    # ---- validation ----
    required_cols = {"model_name", "condition", "solution_count"}
    missing = required_cols - set(df.columns)
    assert not missing, f"df missing required columns: {sorted(missing)}"
    assert metric in df.columns, f"metric '{metric}' not found in df columns."
    if order_model_by_metric is None:
        order_model_by_metric = metric
    assert order_model_by_metric in df.columns, (
        f"order_model_by_metric '{order_model_by_metric}' not found in df columns."
    )

    # ---- filter to one condition ----
    df_cond = df[df["condition"] == condition].copy()
    assert len(df_cond) > 0, f"No rows found for condition='{condition}'."

    # ---- choose models (optionally top-k) ----
    model_rank = (
        df_cond.groupby("model_name")[order_model_by_metric]
        .mean()
        .sort_values(ascending=False)
    )
    models = model_rank.index.tolist()
    assert len(models) > 0, "No models found after filtering."

    if top_k_models is not None:
        models = models[:top_k_models]

    # ---- plot (single axes) ----
    fig, ax = plt.subplots(figsize=(7, 4))

    # stable colors by model count
    cmap = plt.cm.get_cmap("tab10", max(10, len(models)))

    for idx, model in enumerate(models):
        df_m = df_cond[df_cond["model_name"] == model].sort_values("solution_count")

        # in case there are repeated solution_count values, aggregate (mean)
        df_m = (
            df_m.groupby("solution_count", as_index=False)[metric]
            .mean()
            .sort_values("solution_count")
        )

        ax.plot(
            df_m["solution_count"],
            df_m[metric],
            marker="o",
            label=model,
            color=cmap(idx % cmap.N),
        )

    # ax.set_title(f"{metric.replace('_', ' ').title()} vs Solution Count (condition={condition})")
    ax.set_xlabel("# Solutions")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)

    # legend outside if many models
    ncol = 1 if len(models) <= 8 else 2 if len(models) <= 16 else 3
    ax.legend(
        title="Models",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        ncol=ncol,
    )

    fig.tight_layout()

    if save_fp is not None:
        plt.savefig(save_fp, bbox_inches="tight", dpi=300)

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def plot_trend_grid_by_model(
    df: pd.DataFrame,
    *,
    nrows: int = 2,
    model_col: str = "model_name",
    condition_col: str = "condition",
    x_col: str = "solution_count",
    y_col: str = "exact_match_accuracy",
    # plotting controls
    point_size: float = 14,
    mean_lw: float = 2.2,
    mean_marker: str = "o",
    mean_marker_scale: float = 2.2,
    mean_marker_min_size: int = 28,
    mean_marker_alpha: float = 1.0,
    mean_marker_edge: str = "white",
    mean_marker_edge_lw: float = 0.9,
    # layout
    sharey: bool = True,
    figsize_per_subplot: tuple[float, float] = (5.2, 4.6),
    seed: int = 0,
    title: str | None = None,
    save_fp: str | None = None,
    dpi: int = 300,
    y_scale: str = "0-100",  # "auto" | "0-1" | "0-100"
    # color controls
    condition_colors: dict | None = None,
    condition_cmap: str | mpl.colors.Colormap | None = None,
    condition_order: list | None = None,
    sort_conditions: bool = False,
    strict_condition_colors: bool = False,
    # shared legend (above entire figure)
    legend_ncol: int | None = None,
    legend_y: float = -0.05,
    legend_fontsize: float = 12.0,
    legend_title_fontsize: float = 15.0,
    legend_framealpha: float = 0.95,
    legend_loc: str = "lower center",
):
    """
    Completely rewritten (per request):
      - Plot ONLY per-(condition, x) mean trajectories (no CI, no fits).
      - One shared legend above the entire figure.
    """

    def is_binary_series(s: pd.Series) -> bool:
        vals = pd.Series(s.dropna().unique())
        if len(vals) == 0:
            return False
        return set(np.round(vals.astype(float), 10).tolist()).issubset({0.0, 1.0})

    def is_percent_binary_like(s: pd.Series) -> bool:
        vals = pd.Series(s.dropna().unique())
        if len(vals) == 0:
            return False
        return set(np.round(vals.astype(float), 10).tolist()).issubset({0.0, 100.0})

    def _resolve_condition_list(data: pd.DataFrame) -> list:
        if condition_order is not None:
            seen = list(condition_order)
            extra = [c for c in pd.unique(data[condition_col]) if c not in set(seen)]
            return seen + extra
        conds_local = list(pd.unique(data[condition_col]))
        if sort_conditions:
            conds_local = sorted(conds_local, key=lambda x: str(x))
        return conds_local

    def _build_color_map(conds_local: list) -> dict:
        explicit = dict(condition_colors or {})

        missing = [c for c in conds_local if c not in explicit]
        if strict_condition_colors and missing:
            raise ValueError(
                f"strict_condition_colors=True but no colors provided for: {missing}. "
                f"Either add them to condition_colors or set strict_condition_colors=False."
            )

        cmap_out: dict = {c: explicit[c] for c in conds_local if c in explicit}
        remaining = [c for c in conds_local if c not in cmap_out]
        if not remaining:
            return cmap_out

        if condition_cmap is not None:
            cm = plt.get_cmap(condition_cmap) if isinstance(condition_cmap, str) else condition_cmap
            n = len(remaining)
            samples = np.linspace(0.08, 0.92, n) if n > 1 else np.array([0.5])
            colors = [mpl.colors.to_hex(cm(v), keep_alpha=False) for v in samples]
        else:
            palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])
            colors = [palette[i % len(palette)] for i in range(len(remaining))]

        for c, col in zip(remaining, colors):
            cmap_out[c] = col
        return cmap_out

    if nrows < 1:
        raise ValueError("nrows must be >= 1")

    rng = np.random.default_rng(seed)

    data = df.copy()
    req = [model_col, condition_col, x_col, y_col]
    data = data.dropna(subset=req)
    data[x_col] = pd.to_numeric(data[x_col], errors="coerce")
    data[y_col] = pd.to_numeric(data[y_col], errors="coerce")
    data = data.dropna(subset=[x_col, y_col])

    models = list(pd.unique(data[model_col]))
    m = len(models)
    if m == 0:
        raise ValueError("No models found after filtering/cleaning.")

    conds = _resolve_condition_list(data)
    color_map = _build_color_map(conds)

    ncols = math.ceil(m / nrows)
    total_axes = nrows * ncols

    # y scale decision
    if y_scale not in {"auto", "0-1", "0-100"}:
        raise ValueError('y_scale must be one of {"auto","0-1","0-100"}')

    if y_scale == "auto":
        if is_binary_series(data[y_col]):
            y_plot_scale = "0-1"
        elif is_percent_binary_like(data[y_col]):
            y_plot_scale = "0-100"
        else:
            y_plot_scale = "continuous"
    else:
        y_plot_scale = "0-1" if y_scale == "0-1" else "0-100"

    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharey=sharey)
    axes = np.array(axes).reshape(-1)

    mean_marker_size = max(mean_marker_min_size, int(point_size * float(mean_marker_scale)))

    # For shared legend: create one proxy artist per condition (stable, clean)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color_map[c],
            linewidth=mean_lw,
            marker=mean_marker,
            markersize=math.sqrt(mean_marker_size) * 0.35,  # keeps it visually reasonable
            markeredgecolor=mean_marker_edge,
            markeredgewidth=mean_marker_edge_lw,
            label=str(c),
        )
        for c in conds
        if c in color_map
    ]

    for i, model_name in enumerate(models):
        ax = axes[i]
        dmod = data[data[model_col] == model_name].copy()

        for cond in conds:
            d = dmod[dmod[condition_col] == cond].copy()
            if d.empty:
                continue

            color = color_map[cond]

            # per-x mean
            g = d.groupby(x_col, dropna=False)[y_col].mean().reset_index().sort_values(x_col)
            x_vals = g[x_col].to_numpy(dtype=float)
            mean_vals = g[y_col].to_numpy(dtype=float)

            ax.plot(
                x_vals,
                mean_vals,
                color=color,
                linewidth=mean_lw,
                zorder=3,
            )
            ax.scatter(
                x_vals,
                mean_vals,
                marker=mean_marker,
                s=mean_marker_size,
                alpha=mean_marker_alpha,
                color=color,
                edgecolors=mean_marker_edge,
                linewidths=mean_marker_edge_lw,
                zorder=4,
            )

        ax.set_title(str(model_name), fontsize=16)
        ax.set_xlabel("Number of Solutions", fontsize=14)
        ax.grid(True, alpha=0.1)

        if y_plot_scale == "0-100":
            ax.set_ylim(0.0, 100.0)
            ax.set_yticks(list(range(0, 101, 10)))
        elif y_plot_scale == "0-1":
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([k / 10 for k in range(11)])

        ax.tick_params(labelsize=14)

    # turn off unused axes
    for j in range(m, total_axes):
        axes[j].axis("off")

    # y label on left column
    for r in range(nrows):
        axes[r * ncols].set_ylabel(" ".join(y_col.split("_")).title(), fontsize=14)

    # shared legend above figure
    ncol = legend_ncol if legend_ncol is not None else min(len(legend_handles), 5)
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            labels=[h.get_label() for h in legend_handles],
            loc=legend_loc,
            ncol=ncol,
            frameon=True,
            framealpha=legend_framealpha,
            fontsize=legend_fontsize,
            title=condition_col.title(),
            title_fontsize=legend_title_fontsize,
            bbox_to_anchor=(0.5, legend_y),
        )

    # title
    if title is not None:
        fig.suptitle(title, y=legend_y + 0.06)

    # leave room at top for legend (+ title)
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    if save_fp:
        save_fp = Path(save_fp)
        save_fp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fp, dpi=dpi, bbox_inches="tight")

    return fig, axes.reshape(nrows, ncols)
