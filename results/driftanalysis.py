# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 19:39:16 2026

@author: Divya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns

# plt.figure(figsize=(7, 5))

# sns.boxplot(
#     data=df_all,
#     x="source",
#     y="eval_success",
#     showfliers=True,
#     width=0.5
# )

# plt.ylabel("Evaluation Success (%)")
# plt.xlabel("")
# plt.title("Evaluation Success Across Rounds (Box Plot)")
# plt.grid(axis="y", linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(7, 5))

# sns.violinplot(
#     data=df_all,
#     x="source",
#     y="eval_success",
#     inner="quartile",
#     cut=0
# )

# plt.ylabel("Evaluation Success (%)")
# plt.xlabel("")
# plt.title("Evaluation Success Across Rounds (Violin Plot)")
# plt.grid(axis="y", linestyle="--", alpha=0.4)
# plt.tight_layout()
# plt.show()


# fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# sns.boxplot(data=df_all, x="source", y="eval_success", ax=axes[0])
# axes[0].set_title("Box Plot")

# sns.violinplot(
#     data=df_all,
#     x="source",
#     y="eval_success",
#     inner="quartile",
#     cut=0,
#     ax=axes[1]
# )
# axes[1].set_title("Violin Plot")

# for ax in axes:
#     ax.set_xlabel("")
#     ax.set_ylabel("Evaluation Success (%)")
#     ax.grid(axis="y", linestyle="--", alpha=0.4)

# plt.tight_layout()
# plt.show()

from glob import glob




def read_eval_csv(
    csv_path,
    round_col="t",
    agent_col="i",
    success_col="eval_success",
    loss_col="eval_loss",
    drift_col="drift",
    delayed_col="delayed",
):
    """
    Read evaluation CSV into a pandas DataFrame with standard column names.
    """
    df = pd.read_csv(csv_path)

    # Optional: rename to canonical names for consistency
    df = df.rename(columns={
        round_col: "round",
        agent_col: "agent",
        success_col: "eval_success",
        loss_col: "eval_loss",
        drift_col: "drift",
        delayed_col: "delayed",
    })

    # Sort for predictable ordering
    df = df.sort_values(["round", "agent"]).reset_index(drop=True)

    return df


def group_by_round(df):
    """
    Returns:
        dict[int, pd.DataFrame]
        round -> dataframe of all agents in that round
    """
    return {
        r: g.reset_index(drop=True)
        for r, g in df.groupby("round")
    }


def group_by_agent(df):
    """
    Returns:
        dict[int, pd.DataFrame]
        agent -> dataframe of all rounds for that agent
    """
    return {
        a: g.reset_index(drop=True)
        for a, g in df.groupby("agent")
    }




def plot_drift_heatmap(
    df,
    round_col="round",
    agent_col="agent",
    drift_col="drift",
    title="Drift heatmap",
    figsize=None,
    show_colorbar=True,
    xtick_every=None,
    ytick_every=None,
    title_fontsize=16,
    selected_agents=None,        # optional list of sampled agent IDs (length K)
    sampled_order="sorted",      # "sorted" or "first_appearance"
    return_mapping=False,
):
    """
    Heatmap of rounds (x) vs sampled client index (y), colored by drift type.

    Y-axis shows indices 0..K-1 for the sampled set rather than original agent IDs.

    Returns:
      (fig, ax, mat) or (fig, ax, mat, idx_to_agent) if return_mapping=True
    """

    def encode_drift(d):
        if pd.isna(d):
            return 0
        s = str(d).strip().lower()
        if s == "":
            return 0
        # priority: u > cd > cs
        if "u" in s:
            return 3
        if "cd" in s:
            return 2
        if "cs" in s:
            return 1
        return 0

    sns.set(style="whitegrid")

    df2 = df.copy()
    df2["_drift_code"] = df2[drift_col].apply(encode_drift)

    # Normalize agent ids
    df2[agent_col] = pd.to_numeric(df2[agent_col], errors="coerce")
    df2 = df2.dropna(subset=[agent_col])
    df2[agent_col] = df2[agent_col].astype(int)

    # Determine sampled agents
    if selected_agents is None:
        if sampled_order == "first_appearance":
            seen = set()
            selected_agents = []
            for a in df2[agent_col].tolist():
                if a not in seen:
                    seen.add(a)
                    selected_agents.append(a)
        else:
            selected_agents = sorted(df2[agent_col].unique().tolist())

    selected_agents = [int(a) for a in selected_agents]
    K = len(selected_agents)

    # Filter to sampled agents only
    df2 = df2[df2[agent_col].isin(selected_agents)].copy()

    # Remap original agent IDs -> 0..K-1
    agent_to_idx = {aid: i for i, aid in enumerate(selected_agents)}
    idx_to_agent = {i: aid for aid, i in agent_to_idx.items()}

    # ✅ CREATE the column (no parentheses!)
    df2["_agent_idx"] = df2[agent_col].map(agent_to_idx)

    # Pivot to matrix
    mat = df2.pivot(index="_agent_idx", columns=round_col, values="_drift_code")
    mat = mat.sort_index().sort_index(axis=1)
    mat = mat.reindex(range(K))          # ensure exactly K rows
    mat = mat.fillna(0).astype(int)

    # Colormap
    cmap = ListedColormap([
        "#f0f0f0",  # 0: no drift
        "#abd9e9",  # 1: CS
        "#fdae61",  # 2: CD
        "#d7191c",  # 3: U
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # Figure sizing
    if figsize is None:
        figsize = (max(8, 0.35 * mat.shape[1]), max(4, 0.35 * mat.shape[0]))

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        mat.values,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # White gridlines (Seaborn-style)
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Labels / ticks
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Round", fontsize=16)
    ax.set_ylabel("Sampled Client Index", fontsize=16)

    rounds = mat.columns.tolist()
    agents = mat.index.tolist()

    if xtick_every is None:
        xtick_every = max(1, len(rounds) // 20)
    if ytick_every is None:
        ytick_every = 1

    xticks = np.arange(0, len(rounds), xtick_every)
    yticks = np.arange(0, len(agents), ytick_every)

    ax.set_xticks(xticks)
    ax.set_xticklabels([rounds[i] for i in xticks], rotation=90)
    ax.set_yticks(yticks)
    ax.set_yticklabels([agents[i] for i in yticks])

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(["No drift", "CS (low)", "CD (medium)", "U (high)"])

    fig.tight_layout()

    if return_mapping:
        return fig, ax, mat, idx_to_agent

    return fig, ax, mat




def eval_success_stats_from_txt(path):
    """
    Read a CSV-formatted .txt file and compute
    mean and std of eval_success across all rounds t.
    """
    df = pd.read_csv(path)

    mean_success = df["eval_success"].mean()
    std_success = df["eval_success"].std(ddof=1)  # sample std

    return {
        "mean_eval_success": mean_success,
        "std_eval_success": std_success,
        "num_rounds": len(df),
    }



def read_eval_files(file_paths, labels=None):
    """
    Read multiple CSV-formatted .txt files and combine into one DataFrame.

    Args:
        file_paths: list of file paths
        labels: optional list of names (e.g., ['Method A', 'Method B', 'Method C'])

    Returns:
        Combined pandas DataFrame with a 'source' column
    """
    dfs = []

    for idx, path in enumerate(file_paths):
        df = pd.read_csv(path)

        if labels is not None:
            df["source"] = labels[idx]
        else:
            df["source"] = f"Run {idx+1}"

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)



def summarize_eval_file(path):
    """
    File format: columns include at least ['t', 'eval_success'].
    Returns mean and std of eval_success across all rows (e.g., 50 rounds).
    """
    df = pd.read_csv(path)

    if "eval_success" not in df.columns:
        raise ValueError(f"'eval_success' column not found in {path}. Found: {list(df.columns)}")

    mean_val = df["eval_success"].mean()
    std_val  = df["eval_success"].std(ddof=1)  # sample std

    return mean_val, std_val, len(df)



def build_summary_from_files(file_specs):
    """
    file_specs: list of dicts like:
      {
        "path": "path/to/file.txt",
        "imbalance": 0.4,
        "drifted_clients": 4,
        "method": "STROBFL"
      }

    Returns a summary df with columns:
      method, imbalance, drifted_clients, mean, std, n_rounds
    """
    rows = []
    for spec in file_specs:
        mean_val, std_val, n = summarize_eval_file(spec["path"])
        rows.append({
            "method": spec.get("method", "Run"),
            "imbalance": float(spec["imbalance"]),
            "drifted_clients": int(spec["drifted_clients"]),
            "mean": mean_val,
            "std": std_val,
            "n_rounds": n,
            "path": spec["path"],
        })
    return pd.DataFrame(rows)


# file_specs = [
#     {"path": "STROBFL_imb0.0_drift0.txt", "imbalance": 0.0, "drifted_clients": 0, "method": "STROBFL"},
#     {"path": "STROBFL_imb0.4_drift0.txt", "imbalance": 0.4, "drifted_clients": 0, "method": "STROBFL"},
#     {"path": "STROBFL_imb0.8_drift0.txt", "imbalance": 0.8, "drifted_clients": 0, "method": "STROBFL"},
#     {"path": "STROBFL_imb1.0_drift0.txt", "imbalance": 1.0, "drifted_clients": 0, "method": "STROBFL"},

#     {"path": "STROBFL_imb0.0_drift4.txt", "imbalance": 0.0, "drifted_clients": 4, "method": "STROBFL"},
#     # ... add the rest (imb=0.4/0.8/1.0) and drift=8 similarly
# ]
# summary_df = build_summary_from_files(file_specs)

# plot_imbalance_vs_drift_facets(summary_df)

def plot_imbalance_vs_drift_facets(
    summary_df,
    drift_levels=(0, 4, 8),
    imbalance_levels=(0.0, 0.4, 0.8, 1.0),
    title="Global Accuracy vs Imbalance under Different Drift Levels",
    ylabel="Global Accuracy (%)",
):
    """
    summary_df columns: method, imbalance, drifted_clients, mean, std
    """
    methods = list(summary_df["method"].unique())
    n_panels = len(drift_levels)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, drift in zip(axes, drift_levels):
        sub = summary_df[summary_df["drifted_clients"] == drift].copy()

        for method in methods:
            mdf = sub[sub["method"] == method].copy()
            # ensure in imbalance order
            mdf["imbalance"] = mdf["imbalance"].astype(float)
            mdf = mdf.sort_values("imbalance")

            ax.errorbar(
                mdf["imbalance"],
                mdf["mean"],
                yerr=mdf["std"],
                marker="o",
                capsize=4,
                linewidth=2,
                label=method,
            )

        ax.set_title(f"{drift} Drifted Clients")
        ax.set_xlabel("Imbalance Factor")
        ax.set_xticks(list(imbalance_levels))
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel(ylabel)
    axes[-1].legend(title="Method", loc="best", frameon=True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()



file  = "results_data1.csv"
df = read_eval_csv(file,round_col="t",
   agent_col="i",
  success_col="eval_success",
  loss_col="eval_loss",
  drift_col="drift",
   delayed_col="delayed")

group_by_round(df)



plot_drift_heatmap(
    df,
    round_col="round",
    agent_col="agent",
    drift_col="drift",
    title="Drift heatmap",
    title_fontsize=16,
    figsize=None,
    show_colorbar=True,
    xtick_every=None,
    ytick_every=None,
      )
ax = plt.gca()

high_drift = (8, 20)     # e.g., concept drift window
medium_drift    = (30, 42)    # e.g., covariate shift window

def add_band(ax, start, end, color, label, alpha=0.15, linestyle="--"):
    # shade (align to heatmap cell edges)
    ax.axvspan(start - 0.5, end + 0.5, alpha=alpha, color=color, zorder=10)
    # boundaries
    ax.axvline(start - 0.5, linestyle=linestyle, linewidth=1.5, color=color, zorder=11)
    ax.axvline(end + 0.5,   linestyle=linestyle, linewidth=1.5, color=color, zorder=11)
    # label near top of plot
    y0, y1 = ax.get_ylim()
    y_text = min(y0, y1)  # works even if y is inverted
    ax.text(
        x=(start + end) / 2,
        y=y_text,
        s=label,
        color=color,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        zorder=12
    )

add_band(ax, *high_drift, color="red",   label="Drift Area 1")
add_band(ax, *medium_drift,     color="green",  label="Drift Area 2" )
handles, labels = ax.get_legend_handles_labels()
if labels:  # only if something has label=...
    leg = ax.legend(handles, labels, fontsize=16)
    for t in leg.get_texts():
        t.set_fontsize(16)
for txt in ax.texts:
    if txt.get_text() in {"Drift Area 1", "Drift Area 2"}:
        txt.set_fontsize(14)
        txt.set_fontweight("bold")
plt.tight_layout()
plt.show()


drift_start, drift_end = 8, 20

# Shade columns [8,20] (align to cell edges)
ax.axvspan(drift_start - 0.5, drift_end + 0.5, alpha=0.15, color="red", zorder=10)

# Optional: boundary lines at start/end
ax.axvline(drift_start - 0.5, linestyle="--", linewidth=1.5, color="red", zorder=11)
ax.axvline(drift_end + 0.5, linestyle="--", linewidth=1.5, color="red", zorder=11)



# Optional: label at top
y_top = ax.get_ylim()[0]  # note: heatmaps often invert y
ax.text(
    x=(drift_start + drift_end) / 2,
    y=y_top,
    s="Drift window",
    color="red",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    zorder=12,
)


plt.tight_layout()
plt.show()


