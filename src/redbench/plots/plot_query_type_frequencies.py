import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_query_type_frequencies(
    redset_entries: pd.DataFrame,
    wl_df: pd.DataFrame,
    gen_strategy: str,
    artifacts_dir: str = None,
):
    # Bar chart of number of queries per query_type in wl_df and redset_entries
    query_type_counts_wl = wl_df["query_type"].value_counts().sort_index()
    query_type_counts_redset = redset_entries["query_type"].value_counts().sort_index()

    # extract query types
    query_types = sorted(set(wl_df["query_type"].unique()))

    bar_width = 0.35
    x = np.arange(len(query_types))

    # Absolute counts
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Absolute bar plot
    axes[0].bar(
        x - bar_width / 2,
        [query_type_counts_wl.get(q, 0) for q in query_types],
        width=bar_width,
        label=f"Redbench ({gen_strategy})",
        color="tab:blue",
    )
    axes[0].bar(
        x + bar_width / 2,
        [query_type_counts_redset.get(q, 0) for q in query_types],
        width=bar_width,
        label="Redset",
        color="tab:red",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(query_types, rotation=30)
    axes[0].set_ylabel("Number of Queries")
    axes[0].set_title("Absolute Query Type Occurrences")
    axes[0].legend()

    # Normalized bar plot
    total_wl = query_type_counts_wl.sum()
    total_redset = query_type_counts_redset.sum()
    axes[1].bar(
        x - bar_width / 2,
        [query_type_counts_wl.get(q, 0) / total_wl for q in query_types],
        width=bar_width,
        label=f"Redbench ({gen_strategy})",
        color="tab:blue",
    )
    axes[1].bar(
        x + bar_width / 2,
        [query_type_counts_redset.get(q, 0) / total_redset for q in query_types],
        width=bar_width,
        label="Redset",
        color="tab:red",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(query_types, rotation=30)
    axes[1].set_ylabel("Fraction of Queries")
    axes[1].set_title("Relative Query Type Occurrences")
    axes[1].legend()

    plt.tight_layout()

    if artifacts_dir:
        output_path = os.path.join(artifacts_dir, "query_type_distribution.png")
        plt.savefig(output_path)
    else:
        plt.show()

    ###
    # plot num joins and num scans
    ###
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Plot number of joins
    if "num_joins" in wl_df.columns and "num_joins" in redset_entries.columns:
        max_joins = max(wl_df["num_joins"].max(), redset_entries["num_joins"].max())
        bins_joins = np.arange(0, max_joins + 2) - 0.5
        axes[0].hist(
            redset_entries["num_joins"],
            bins=bins_joins,
            alpha=0.7,
            label="Redset",
            color="tab:red",
        )
        axes[0].hist(
            wl_df["num_joins"],
            bins=bins_joins,
            alpha=0.7,
            label=f"Redbench ({gen_strategy})",
            color="tab:blue",
        )
        axes[0].set_xlabel("Number of Joins")
        axes[0].set_ylabel("Number of Queries")
        axes[0].set_title("Distribution of Number of Joins")
        axes[0].legend()

    # Plot number of scans
    if "num_scans" in wl_df.columns and "num_scans" in redset_entries.columns:
        max_scans = max(wl_df["num_scans"].max(), redset_entries["num_scans"].max())
        bins_scans = np.arange(0, max_scans + 2) - 0.5
        axes[1].hist(
            redset_entries["num_scans"],
            bins=bins_scans,
            alpha=0.7,
            label="Redset",
            color="tab:red",
        )
        axes[1].hist(
            wl_df["num_scans"],
            bins=bins_scans,
            alpha=0.7,
            label=f"Redbench ({gen_strategy})",
            color="tab:blue",
        )
        axes[1].set_xlabel("Number of Scans")
        axes[1].set_ylabel("Number of Queries")
        axes[1].set_title("Distribution of Number of Scans")
        axes[1].legend()

    plt.tight_layout()

    if artifacts_dir:
        output_path = os.path.join(artifacts_dir, "num_joins_scans_distribution.png")
        plt.savefig(output_path)
    else:
        plt.show()
