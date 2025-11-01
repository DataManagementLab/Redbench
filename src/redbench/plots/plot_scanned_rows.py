import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_scanned_rows(
    redset_entries: pd.DataFrame,
    wl_df: pd.DataFrame,
    artifacts_dir: str = None,
    query_type="all",
):
    local_redset_entries = redset_entries.copy()
    local_wl_df = wl_df.copy()

    # filter by query type if specified
    if query_type != "all":
        local_redset_entries = local_redset_entries[
            local_redset_entries["query_type"] == query_type
        ]
        local_wl_df = local_wl_df[local_wl_df["query_type"] == query_type]

    redset_agg = local_redset_entries.groupby(
        local_redset_entries["arrival_timestamp"].dt.floor("h")
    )["mbytes_scanned"].sum()
    wl_agg = local_wl_df.groupby(local_wl_df["arrival_timestamp"].dt.floor("h"))[
        "rows_scanned"
    ].sum()

    def plot(prefix_sum: bool):
        title = "Rows Scanned (wl_gen) and MBytes Scanned (redset) Over Time"
        if prefix_sum:
            title = "Cumulative " + title

        if query_type != "all":
            title = f"{title}   [{query_type}]"

        # Use cumulative sum if prefix_sum is True
        redset_data = redset_agg.cumsum() if prefix_sum else redset_agg
        wl_data = wl_agg.cumsum() if prefix_sum else wl_agg

        fig, ax1 = plt.subplots(figsize=(12, 4))
        color1 = "tab:blue"
        color2 = "tab:red"

        ax1.set_ylabel("MBytes Scanned (Redset)", color=color2)
        ax1.plot(
            redset_data.index,
            redset_data.values,
            label="Redset: mbytes_scanned",
            color=color2,
        )
        ax1.tick_params(axis="y", labelcolor=color2)

        ax2 = ax1.twinx()
        ax2.set_xlabel("Arrival Time")
        ax2.set_ylabel("Rows Scanned (Redbench)", color=color1)
        ax2.plot(
            wl_data.index,
            wl_data.values,
            label="Redbench: rows_scanned",
            color=color1,
            alpha=0.7,
        )
        ax2.tick_params(axis="y", labelcolor=color1)

        plt.title(title)
        fig.tight_layout()
        if artifacts_dir:
            suffix = "_cumulative" if prefix_sum else ""
            output_path = os.path.join(
                artifacts_dir, f"scanned_rows_{query_type}{suffix}.png"
            )
            plt.savefig(output_path)
        else:
            plt.show()

    plot(prefix_sum=False)
    plot(prefix_sum=True)

    # Compute scale factor based on max values
    if not wl_agg.empty and not redset_agg.empty:
        max_rows = wl_agg.max()
        max_mbytes = redset_agg.max()
        if max_rows > 0:
            scale_rows = max_mbytes / max_rows
        else:
            scale_rows = 1.0
        wl_agg_scaled = wl_agg * scale_rows

        # Align indexes for diff
        common_idx = wl_agg_scaled.index.intersection(redset_agg.index)
        diff = wl_agg_scaled[common_idx] - redset_agg[common_idx]

        plt.close()  # Close any existing plots
        plt.figure(figsize=(12, 2))
        plt.plot(common_idx, diff, linestyle="-", color="purple")
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        title = "Difference: Scaled Rows Scanned (Redbench) - MBytes Scanned (Redset)"
        if query_type != "all":
            title = f"{title}   [{query_type}]"

        plt.title(title)
        plt.xlabel("Arrival Time")
        plt.ylabel("Scaled Rows - MBytes")
        plt.tight_layout()

        if artifacts_dir:
            output_path = os.path.join(
                artifacts_dir, f"scanned_rows_{query_type}_diff.png"
            )
            plt.savefig(output_path)
        else:
            plt.show()
    else:
        print("One of the series is empty, cannot compute diff.")
