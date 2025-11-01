import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Load the CSV and plot
def plot_repetition(
    redset_entries: pd.DataFrame, wl_df: pd.DataFrame, artifacts_dir: str = None
):
    show_structural = "structural_repetition_id" in wl_df.columns

    # extract num query-templates and num unique queries
    if show_structural:
        num_query_templates = wl_df.groupby("query_type")[
            "structural_repetition_id"
        ].nunique()
    else:
        num_query_templates = None
    num_unique_queries = wl_df.groupby("query_type")["sql"].nunique()

    # preprocess redset entries to count repetitions
    # compute an exact repetition hash for redset entries
    redset_entries["exact_repetition_hash"] = redset_entries.apply(
        lambda row: hash(
            (
                row["query_type"],
                row["num_joins"],
                row["num_aggregations"],
                row["read_table_ids"],
                row["write_table_ids"],
                row["feature_fingerprint"],
                row["database_id"],
                row["instance_id"],
            )
        ),
        axis=1,
    )

    # count repetitions
    if show_structural:
        count_structural_repetitions = (
            wl_df.groupby(["query_type", "structural_repetition_id"])
            .size()
            .reset_index(name="count")
        )
        count_exact_repetitions = (
            wl_df.groupby(["query_type", "structural_repetition_id", "sql"])
            .size()
            .reset_index(name="count")
        )
    else:
        count_exact_repetitions = (
            wl_df.groupby(["query_type", "sql"]).size().reset_index(name="count")
        )
    count_exact_repetitions_redset = (
        redset_entries.groupby(["query_type", "exact_repetition_hash"])
        .size()
        .reset_index(name="count")
    )

    # filter out query types from redset that are not in the workload
    count_exact_repetitions_redset = count_exact_repetitions_redset[
        count_exact_repetitions_redset["query_type"].isin(num_unique_queries.index)
    ]

    ###
    # Num unique queries barchart
    ###

    # Prepare data for bar chart
    query_types = num_unique_queries.index
    if show_structural:
        df_bar = pd.DataFrame(
            {
                "query_type": query_types,
                "Num Query Templates": num_query_templates.values,
                "Num Unique Queries": num_unique_queries.values,
            }
        ).melt(id_vars="query_type", var_name="Metric", value_name="Count")
    else:
        df_bar = pd.DataFrame(
            {
                "query_type": query_types,
                "Num Unique Queries": num_unique_queries.values,
            }
        ).melt(id_vars="query_type", var_name="Metric", value_name="Count")

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_bar, x="query_type", y="Count", hue="Metric")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Query Type")
    plt.ylabel("Count")
    if show_structural:
        plt.title("Number of Query Templates and Unique Queries per Query Type")
    else:
        plt.title("Number of Unique Queries per Query Type")
    plt.legend(title="Metric")
    plt.tight_layout()
    if artifacts_dir:
        output_path = f"{artifacts_dir}/num_queries_templates_per_type.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

    ###
    # Plot Mean count of repetitions per query type
    ###

    # Calculate mean count for each query_type
    if show_structural:
        mean_structural_repetitions = (
            count_structural_repetitions.groupby("query_type")["count"]
            .mean()
            .reset_index()
        )
        mean_structural_repetitions["hash_type"] = "Structural"

    mean_exact_repetitions = (
        count_exact_repetitions.groupby("query_type")["count"].mean().reset_index()
    )
    mean_exact_repetitions["hash_type"] = "Exact"

    mean_exact_repetitions_redset = (
        count_exact_repetitions_redset.groupby("query_type")["count"]
        .mean()
        .reset_index()
    )
    mean_exact_repetitions_redset["hash_type"] = "Exact (Redset)"

    # Combine dataframes
    if show_structural:
        mean_counts = pd.concat(
            [
                mean_structural_repetitions,
                mean_exact_repetitions,
                mean_exact_repetitions_redset,
            ]
        )
    else:
        mean_counts = pd.concat([mean_exact_repetitions, mean_exact_repetitions_redset])

    # Plot
    plt.close()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mean_counts, x="query_type", y="count", hue="hash_type")
    plt.yscale("log")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Query Type")
    plt.ylabel("Mean Count")
    if show_structural:
        plt.title("Mean Count of Queries per Query Type (Structural vs Exact)")
    else:
        plt.title("Mean Count of Queries per Query Type (Exact)")
    plt.legend(title="Hash Type")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    if artifacts_dir:
        plt.tight_layout()
        output_path = f"{artifacts_dir}/query_repetitions.png"
        plt.savefig(output_path)
    else:
        plt.show()
    ###
    # Multiplot: Histogram of exact and structural repetitions for each query type
    ###
    query_types = wl_df["query_type"].unique()
    n_types = len(query_types)
    ncols = 3
    nrows = (n_types + ncols - 1) // ncols

    # Find global min and max for x-axis
    if show_structural:
        all_struct_counts = count_structural_repetitions["count"]
        all_exact_counts = count_exact_repetitions["count"]
        global_min = min(all_struct_counts.min(), all_exact_counts.min())
        global_max = max(all_struct_counts.max(), all_exact_counts.max())
    else:
        all_exact_counts = count_exact_repetitions["count"]
        global_min = all_exact_counts.min()
        global_max = all_exact_counts.max()

    # Define log-spaced bins for all histograms
    num_bins = 30  # You can adjust the number of bins as needed
    # Avoid log(0) by starting from at least 1
    min_bin = max(1, global_min)
    bins = np.logspace(np.log10(min_bin), np.log10(global_max), num_bins + 1)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
    )
    for idx, query_type in enumerate(query_types):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        # Exact repetition counts
        exact_counts = count_exact_repetitions[
            count_exact_repetitions["query_type"] == query_type
        ]["count"]

        # redset exact counts
        redset_exact_counts = count_exact_repetitions_redset[
            count_exact_repetitions_redset["query_type"] == query_type
        ]["count"]

        ax.hist(
            redset_exact_counts,
            bins=bins,
            alpha=0.7,
            label="Exact (Redset)",
            color="tab:green",
        )

        if show_structural:
            # Structural repetition counts
            struct_counts = count_structural_repetitions[
                count_structural_repetitions["query_type"] == query_type
            ]["count"]
            ax.hist(
                struct_counts,
                bins=bins,
                alpha=0.7,
                label="Structural",
                color="tab:blue",
            )

        ax.hist(exact_counts, bins=bins, alpha=0.7, label="Exact", color="tab:orange")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(min_bin, global_max)
        ax.set_ylim(
            0.75, ax.get_ylim()[1] * 1.1
        )  # Extend y-axis a bit for better visibility
        ax.set_title(query_type)
        ax.set_xlabel("Number of Repetitions (log scale)")
        ax.set_ylabel("Frequency")
        ax.legend()

    # Remove empty subplots
    for idx in range(n_types, nrows * ncols):
        fig.delaxes(axes[idx // ncols][idx % ncols])

    plt.tight_layout()
    if artifacts_dir:
        output_path = f"{artifacts_dir}/query_repetitions_hist.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

    ##
    # Boxplot: Distribution of repetition counts for each query type (Structural vs Exact)
    ###

    # Prepare data for boxplot
    if show_structural:
        struct_box = count_structural_repetitions.rename(
            columns={"count": "Repetition Count"}
        )
        struct_box["Repetition Type"] = "Structural"

    exact_box = count_exact_repetitions.rename(columns={"count": "Repetition Count"})
    exact_box["Repetition Type"] = "Exact"
    exact_box_redset = count_exact_repetitions_redset.rename(
        columns={"count": "Repetition Count"}
    )
    exact_box_redset["Repetition Type"] = "Exact (Redset)"

    # For consistency, keep only the relevant columns
    if show_structural:
        struct_box = struct_box[["query_type", "Repetition Count", "Repetition Type"]]
    exact_box = exact_box[["query_type", "Repetition Count", "Repetition Type"]]
    exact_box_redset = exact_box_redset[
        ["query_type", "Repetition Count", "Repetition Type"]
    ]

    if show_structural:
        boxplot_df = pd.concat(
            [struct_box, exact_box, exact_box_redset], ignore_index=True
        )
    else:
        boxplot_df = pd.concat([exact_box, exact_box_redset], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=boxplot_df, x="query_type", y="Repetition Count", hue="Repetition Type"
    )
    plt.yscale("log")
    plt.xlabel("Query Type")
    plt.ylabel("Repetition Count (log scale)")
    plt.title("Distribution of Repetition Counts per Query Type")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Repetition Type")
    plt.tight_layout()
    if artifacts_dir:
        output_path = f"{artifacts_dir}/query_repetitions_boxplot.png"
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
