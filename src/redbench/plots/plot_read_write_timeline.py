import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_read_write_timeline(
    artifacts_dict: Dict,
    log_str: str = "",
    cluster_id: int = None,
    db_id: int = None,
    plot_dir: str = None,
):
    """
    Plot read/write timeline for multiple runs from artifacts_dict.

    Args:
        artifacts_dict: Dictionary containing cluster_id -> db_id -> series_name -> (metadata, df)
        log_str: Optional string to append to output filename
        cluster_id: Optional specific cluster to plot. If None, plots first available cluster.
        db_id: Optional specific database to plot. If None, plots first available db in cluster.
    """
    plt.close()

    # Helper to classify queries as read or write
    def classify_query_type(df):
        write_keywords = ["update", "delete", "insert", "copy", "ctas"]

        def is_write(qtype):
            qtype_lower = str(qtype).lower()
            return any(kw in qtype_lower for kw in write_keywords)

        return df["query_type"].apply(lambda x: "write" if is_write(x) else "read")

    # Get cluster and db to plot
    if cluster_id is None:
        cluster_id = list(artifacts_dict.keys())[0]
    if db_id is None:
        db_id = list(artifacts_dict[cluster_id].keys())[0]

    series_dict = artifacts_dict[cluster_id][db_id]

    # Get all series names and sort them (redset first)
    series_names = sorted(series_dict.keys())
    if "redset" in series_names:
        series_names = ["redset"] + [name for name in series_names if name != "redset"]

    # Define series name lookup and colors
    series_names_lookup = {
        "generation": "Redbench\n(Generation)",
        "matching": "Redbench\n(Matching)",
        "baseline_round_robin": "Round Robin\n(Baseline)",
        "redset": "Redset",
    }

    color_dict = {
        "generation": "#1f77b4",  # blue
        "matching": "#ff7f0e",  # orange
        "redset": "#fe4b4b",  # red
        "baseline_round_robin": "#2ca02c",  # green
    }

    write_color_cache: Dict[str, str] = {}

    def darker_color(hex_color: str, factor: float = 0.85) -> str:
        hex_color = hex_color.lstrip("#")
        rgb = [int(hex_color[i : i + 2], 16) for i in range(0, 6, 2)]
        darker_rgb = [max(0, min(255, int(channel * factor))) for channel in rgb]
        return "#{:02x}{:02x}{:02x}".format(*darker_rgb)

    # Calculate figure height based on number of series
    n_series = len(series_names)
    fig_height = max(2, n_series * 0.6)

    plt.figure(figsize=(14, fig_height))

    markers = {
        "read": "|",
        "write": "|",
    }

    # Prepare datasets
    datasets = []
    for series_name in series_names:
        df = series_dict[series_name][1]
        label = series_names_lookup.get(series_name, series_name)
        datasets.append((df, label, series_name))

    # sort the datasets: redset, redbench, baseline
    datasets.sort(
        key=lambda x: [
            "redset",
            "generation",
            "matching",
            "baseline_round_robin",
        ].index(x[2]),
        reverse=True,
    )

    # Plot each series
    yticks_positions = []
    yticks_labels = []
    sep_lines = []
    label_positions = []
    label_texts = []

    row_height = 0.1
    for idx, (df, label, series_name) in enumerate(datasets):
        df = df.copy()
        df["rw_type"] = classify_query_type(df)

        # y positions for read/write
        base_y = idx * row_height
        read_y = base_y
        write_y = base_y + 0.03

        # For drilldown: only show 'Read' and 'Write' as ytick labels, and highlight main label between them
        yticks_positions.extend([read_y, write_y])
        yticks_labels.extend(["SELECT", "DML"])

        # Store position for main label (centered between read and write)
        label_positions.append((read_y + write_y) / 2)
        label_texts.append(label)

        for rw_type in ["read", "write"]:
            sub = df[df["rw_type"] == rw_type]
            if len(sub) == 0:
                continue

            y_pos = write_y if rw_type == "write" else read_y
            y = np.full(len(sub), y_pos)

            # All reads green, all writes red
            # color = "#2ca02c" if rw_type == "read" else "#fe4b4b"
            color = color_dict.get(series_name, "#000000")

            if rw_type == "write":
                if color not in write_color_cache:
                    write_color_cache[color] = darker_color(color, factor=1.5)
                color = write_color_cache[color]

            alpha = 0.8 if rw_type == "write" else 0.5

            plt.scatter(
                sub["arrival_timestamp"],
                y,
                marker=markers[rw_type],
                color=color,
                s=50,
                alpha=alpha,
            )

        plt.xticks(fontname="monospace")

        # Add a horizontal separator line after each approach
        sep_y = base_y + row_height
        sep_lines.append(sep_y)

    # Draw horizontal separator lines between approaches
    for sep_y in sep_lines[:-1]:
        plt.axhline(sep_y - 0.04, color="#888", linestyle="--", linewidth=1, alpha=0.5)

    plt.yticks(yticks_positions, yticks_labels)

    # Highlight main label for each series (centered between read/write)
    for pos, text in zip(label_positions, label_texts):
        plt.text(
            -0.06,  # move further left to avoid overlap
            pos,
            text,
            va="center",
            ha="right",
            fontweight="bold",
            fontsize=11,
            transform=plt.gca().get_yaxis_transform(),
        )

    plt.xlabel("Arrival Time")
    # plt.title(
    #     f"Read/Write Timeline - Cluster {cluster_id}, DB {db_id}", fontweight="bold"
    # )
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        output_path = os.path.join(plot_dir, "read_write_timeline")
    else:
        os.makedirs("output", exist_ok=True)
        output_path = f"output/read_write_timeline_c{cluster_id}_db{db_id}{log_str}"
    plt.savefig(output_path + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(output_path + ".pdf", dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
