import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_aggregated_arrival_times(
    aggregate_by: str,
    redset_entries: pd.DataFrame,
    wl_df: pd.DataFrame,
    gen_strategy: str,
    num_queries_scale_factor: float = 1.0,
    artifacts_dir: str = None,
):
    """
    Plot arrival times aggregated by either 'count' (number of queries) or 'num_joins' (sum of joins).
    """

    assert aggregate_by in ["count", "num_joins", "num_scans"], (
        f"Invalid aggregate_by value: {aggregate_by}. Must be one of ['count', 'num_joins', 'num_scans']."
    )

    local_redset_entries = redset_entries.copy()
    local_wl_df = wl_df.copy()

    # Group by arrival time (rounded to hour for clarity)
    if aggregate_by == "count":
        redset_agg = local_redset_entries.groupby(
            local_redset_entries["arrival_timestamp"].dt.floor("h")
        ).size()
        wl_agg = local_wl_df.groupby(
            local_wl_df["arrival_timestamp"].dt.floor("h")
        ).size()
        ylabel = "Number of Queries"
        title = "Arrival Times of Queries"
    else:
        assert aggregate_by in local_redset_entries.columns, (
            f"Column '{aggregate_by}' not found in redset entries."
        )
        assert aggregate_by in local_wl_df.columns, (
            f"Column '{aggregate_by}' not found in workload dataframe."
        )

        redset_agg = local_redset_entries.groupby(
            local_redset_entries["arrival_timestamp"].dt.floor("h")
        )[aggregate_by].mean()
        wl_agg = local_wl_df.groupby(local_wl_df["arrival_timestamp"].dt.floor("h"))[
            aggregate_by
        ].mean()
        ylabel = f"Sum of {aggregate_by.capitalize()}"
        title = f"Aggregate {aggregate_by.capitalize()} Over Time"

    # Scale the workload aggregate by the scale factor
    if aggregate_by == "count":
        wl_agg = wl_agg * num_queries_scale_factor

    if aggregate_by == "num_joins":
        y_max = 10
    elif aggregate_by == "num_scans":
        y_max = 10
    else:
        y_max = None

    plt.close()  # Close any existing plots
    plt.figure(figsize=(12, 4))
    plt.plot(redset_agg.index, redset_agg.values, label="Redset", color="tab:red")
    plt.plot(
        wl_agg.index,
        wl_agg.values,
        label=f"Redbench ({gen_strategy})",
        color="tab:blue",
        alpha=0.7,
    )
    plt.xlabel("Arrival Time")
    plt.ylabel(ylabel)
    plt.title(title)

    if y_max is not None:
        plt.ylim(-0.5, y_max)

    plt.legend()
    plt.tight_layout()

    if artifacts_dir:
        output_path = os.path.join(
            artifacts_dir, f"redset_comparison_{aggregate_by}.png"
        )
        plt.savefig(output_path)
    else:
        plt.show()


def plot_arrival_time_by_query_type(
    redset_entries: pd.DataFrame,
    wl_df: pd.DataFrame,
    gen_strategy: str,
    scale_factor: float = 1.0,
    artifacts_dir: str = None,
):
    local_redset_entries = redset_entries.copy()
    local_wl_df = wl_df.copy()

    query_types = sorted(set(local_wl_df["query_type"].unique()))

    def plot(prefix_sum: bool):
        nrows = 2
        ncols = int(np.ceil(len(query_types) / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows), sharex=True)
        axes = np.array(axes).flatten()  # Always flatten to 1D array

        for i, qtype in enumerate(query_types):
            ax = axes[i]
            assert not isinstance(ax, np.ndarray), (
                f"Expected ax to be a single Axes instance, got: {type(ax)} \n {query_types} \n {axes}"
            )
            # Redset
            redset_q = local_redset_entries[local_redset_entries["query_type"] == qtype]
            redset_agg = redset_q.groupby(
                redset_q["arrival_timestamp"].dt.floor("h")
            ).size()
            # Workload
            wl_q = local_wl_df[local_wl_df["query_type"] == qtype]
            wl_agg = (
                wl_q.groupby(wl_q["arrival_timestamp"].dt.floor("h")).size()
                * scale_factor
            )

            if prefix_sum:
                redset_agg = redset_agg.cumsum()
                wl_agg = wl_agg.cumsum()

            # Plot both series
            ax.plot(
                redset_agg.index, redset_agg.values, label="Redset", color="tab:red"
            )
            ax.plot(
                wl_agg.index,
                wl_agg.values,
                label=f"Redbench ({gen_strategy})",
                color="tab:blue",
                alpha=0.7,
            )
            ax.set_ylabel(
                "Cumulative Number of Queries" if prefix_sum else "Number of Queries"
            )
            ax.set_title(f"{qtype}")
            ax.legend()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_tick_params(rotation=30)

        # Hide unused axes if any
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.xlabel("Arrival Time")
        plt.tight_layout()

        if artifacts_dir:
            if prefix_sum:
                output_path = os.path.join(
                    artifacts_dir, "arrival_time_by_query_type_prefix_sum.png"
                )
            else:
                output_path = os.path.join(
                    artifacts_dir, "arrival_time_by_query_type.png"
                )
            plt.savefig(output_path)
        else:
            plt.show()

    plot(prefix_sum=False)
    plot(prefix_sum=True)
