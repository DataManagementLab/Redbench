import itertools
import math
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from utils.load_and_preprocess_redset import get_scanset_from_redset_query


def plot_cluster_overview(artifacts_dict, log_str: str):
    # # _plot_cluster_grid(artifacts_dict, aggregate_by="num_joins", log_str=log_str)
    # # _plot_cluster_grid(artifacts_dict, aggregate_by="num_scans", log_str=log_str)
    # _plot_cluster_grid(artifacts_dict, aggregate_by="count", log_str=log_str)
    # # _plot_cluster_grid(artifacts_dict, aggregate_by="runtime", log_str=log_str)
    # _plot_query_type_by_cluster(artifacts_dict, log_str=log_str)
    # _plot_query_repetition_ratios(artifacts_dict, log_str=log_str, only_w_dml=True, on_first1k=False)
    _plot_query_repetition_ratios(
        artifacts_dict, log_str=log_str, only_w_dml=False, on_first1k=True
    )
    # plot_read_write_timeline(artifacts_dict, log_str=log_str)


def _draw_unmatched(
    ax,
    query_type_counts_dict: Dict,
    sorted_cluster_db_ids: List,
    query_types_order: List,
    norm=False,
):
    bar_width = 0.9
    nesting_margin = 0.05
    for series_ctr, (series_name, cluster_counts) in enumerate(
        query_type_counts_dict.items()
    ):
        if series_name == "redset":
            continue

        if "matching" in series_name:
            gen_strategy = "matching"
        elif "generation" in series_name:
            gen_strategy = "generation"
        elif "baseline" in series_name:
            # we do not visualize unmatched for baseline
            continue
        else:
            raise ValueError(f"Unknown series name: {series_name}")

        for cluster_idx, cluster_db_id in enumerate(sorted_cluster_db_ids):
            redset_counts = query_type_counts_dict["redset"][cluster_db_id]
            series_counts = cluster_counts.get(cluster_db_id, defaultdict(int))
            redset_total = sum(
                redset_counts.get(qt.lower(), 0) for qt in query_types_order
            )
            series_total = (
                sum(series_counts.get(qt.lower(), 0) for qt in query_types_order) or 1
            )
            for qt_idx, qt in enumerate(query_types_order):
                redset_count = redset_counts.get(qt.lower(), 0)
                series_count = series_counts.get(qt.lower(), 0)
                if norm:
                    redset_val = redset_count / redset_total if redset_total else 0
                    series_val = series_count / series_total if series_total else 0
                else:
                    redset_val = redset_count
                    series_val = series_count

                if series_val < redset_val:
                    unmatched = redset_val - series_val

                    if gen_strategy == "matching":
                        edgecolor = "black"
                        hatch = "//"
                    elif gen_strategy == "generation":
                        edgecolor = "gold"
                        hatch = "\\\\"
                    else:
                        raise ValueError(f"Unknown generation strategy: {gen_strategy}")

                    ax.add_patch(
                        plt.Rectangle(
                            (
                                cluster_idx
                                - bar_width / 2
                                + series_ctr * nesting_margin,
                                sum(
                                    (
                                        redset_counts.get(q.lower(), 0) / redset_total
                                        if norm and redset_total
                                        else redset_counts.get(q.lower(), 0)
                                    )
                                    for q in query_types_order[:qt_idx]
                                ),
                            ),
                            bar_width - series_ctr * 2 * nesting_margin,
                            unmatched,
                            fill=False,
                            edgecolor=edgecolor,
                            linestyle="dashed",
                            linewidth=1,
                            hatch=hatch,
                            label=f"unmatched ({series_name})"
                            if f"unmatched ({series_name})"
                            not in [p.get_label() for p in ax.patches]
                            else None,
                        )
                    )


def _plot_query_type_by_cluster(
    artifacts_dict: Dict, log_str: str = "", plot_width: int = None
):
    # Aggregate query types for 'redset' across all clusters
    plt.close()
    query_type_counts_dict = defaultdict(dict)

    # create cluster/db id tuples
    cluster_db_tuples = [
        (cluster_id, db_id)
        for cluster_id, db_dict in artifacts_dict.items()
        for db_id in db_dict.keys()
    ]
    cluster_db_tuples.sort()  # sort for consistent plotting

    for cluster_id, db_id in cluster_db_tuples:
        series_dict = artifacts_dict[cluster_id][db_id]
        for series_name, series_data in series_dict.items():
            df = series_data[1]
            if "query_type" not in df.columns:
                continue
            counts = df["query_type"].value_counts()
            query_type_counts_dict[series_name][(cluster_id, db_id)] = counts

    # Setup
    query_types_order = ["SELECT", "INSERT", "UPDATE", "DELETE"]
    colors = {
        "SELECT": "#3377b6",
        "INSERT": "#40a020",
        "UPDATE": "#9168bf",
        "DELETE": "#dc77c4",
        "CTAS": "#ff7f0e",
    }
    sorted_cluster_db_ids = sorted(
        query_type_counts_dict["redset"].keys(),
        key=lambda cid: query_type_counts_dict["redset"][cid].sum(),
        reverse=True,
    )

    def build_data(norm=False):
        data = []
        for cluster_db_id in sorted_cluster_db_ids:
            counts = query_type_counts_dict["redset"][cluster_db_id]
            row = [counts.get(qt.lower(), 0) for qt in query_types_order]
            data.append(row)
        df = pd.DataFrame(data, columns=query_types_order, index=sorted_cluster_db_ids)
        if norm:
            df = df.div(df.sum(axis=1), axis=0).fillna(0)
        return df

    # Plot
    fig, axes = plt.subplots(
        1, 2, figsize=(8 if plot_width is None else plot_width, 3), sharey=False
    )
    bar_width = 0.9

    # Absolute
    df_plot = build_data(norm=False)
    ax = df_plot.plot(
        kind="bar",
        stacked=True,
        color=[colors[qt] for qt in query_types_order],
        width=bar_width,
        ax=axes[0],
        legend=False,
    )
    _draw_unmatched(
        ax, query_type_counts_dict, sorted_cluster_db_ids, query_types_order, norm=False
    )
    ax.set_xlabel("Cluster/DB ID")
    ax.set_ylabel("# Queries")
    ax.set_title("Query Types per Cluster (Absolute)", fontweight="bold")

    # convert xticks to string cluster/db id
    ax.set_xticklabels([f"{cluster_id}/{db_id}" for cluster_id, db_id in df_plot.index])

    ax.tick_params(axis="x", rotation=90)
    for label in ax.get_xticklabels():
        label.set_fontname("DejaVu Serif")
        # label.set_fontstyle("italic")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)

    # Normalized
    df_plot_norm = build_data(norm=True)
    ax2 = df_plot_norm.plot(
        kind="bar",
        stacked=True,
        color=[colors[qt] for qt in query_types_order],
        width=bar_width,
        ax=axes[1],
        legend=False,
    )
    _draw_unmatched(
        ax2, query_type_counts_dict, sorted_cluster_db_ids, query_types_order, norm=True
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel(
        "Fraction of Queries",
    )
    ax2.set_title("Query Types per Cluster (Normalized)", fontweight="bold")
    ax2.tick_params(axis="x", rotation=90)
    # Set font properties for x-tick labels
    for label in ax2.get_xticklabels():
        label.set_fontname("DejaVu Serif")
        # label.set_fontstyle("italic")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax2.set_ylim(0, 1.0)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[qt]) for qt in query_types_order
    ]
    labels = [qt.title() for qt in query_types_order]
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="black",
            linestyle="dashed",
            linewidth=1,
            hatch="//",
        )
    )
    labels.append("Unmatched (Matching)")
    handles.append(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="gold",
            linestyle="dashed",
            linewidth=1,
            hatch="//",
        )
    )
    labels.append("Unmatched (Generation)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.55, 0.01),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("output", exist_ok=True)
    plt.savefig(
        f"output/cluster_overview_query_type_multiplot{log_str}.png",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_cluster_grid(artifacts_dict: Dict, aggregate_by: str, log_str: str = ""):
    """
    Plot a grid of cluster overview plots.
    Each cluster gets its own subplot (database_id is ignored).
    Each series within a cluster is plotted as a separate line.
    """
    plt.close()
    assert aggregate_by in ["count", "num_joins", "num_scans", "runtime"], (
        f"Invalid aggregate_by value: {aggregate_by}. Must be one of ['count', 'num_joins', 'num_scans', 'runtime']."
    )

    if aggregate_by == "runtime":
        aggregate_by = "execution_duration_ms"

    # create cluster/db id tuples
    cluster_db_tuples = [
        (cluster_id, db_id)
        for cluster_id, db_dict in artifacts_dict.items()
        for db_id in db_dict.keys()
    ]
    cluster_db_tuples.sort()  # sort for consistent plotting
    n_experiments = len(cluster_db_tuples)

    n_cols = 4
    n_rows = math.ceil(n_experiments / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    series_names = sorted(
        artifacts_dict[cluster_db_tuples[0][0]][cluster_db_tuples[0][1]].keys()
    )

    # put redset first in the list
    if "redset" in series_names:
        series_names = ["redset"] + [name for name in series_names if name != "redset"]

    # Determine the global x-axis range
    all_dates = []
    for cluster_id, db_id in cluster_db_tuples:
        series_dict = artifacts_dict[cluster_id][db_id]
        for series_name in series_names:
            if series_name in series_dict:
                df = series_dict[series_name][1]
                all_dates.extend(df["arrival_timestamp"])
    global_min_date = min(all_dates).floor("D")
    global_max_date = max(all_dates).ceil("D")

    for idx, (cluster_id, db_id) in enumerate(cluster_db_tuples):
        ax = axes[idx]
        series_dict = artifacts_dict[cluster_id][db_id]

        # Define color palette, always use grey for 'redset'
        # Use a different color palette, e.g., 'tab10'
        color_cycle = plt.get_cmap("tab10").colors
        color_map = {}
        other_series = [name for name in series_names if name != "redset"]
        for name, color in zip(other_series, itertools.cycle(color_cycle[0:])):
            color_map[name] = color
        color_map["redset"] = "#FF0000"  # red

        for series_name in series_names:
            if series_name not in series_dict:
                print(
                    f"Warning: Series '{series_name}' not found in cluster {cluster_id}."
                )
                continue

            df = series_dict[series_name][1].fillna(0)

            # Assume data is a DataFrame or dict-like with 'arrival_timestamp' and aggregate_by columns
            aggregate_col = "arrival_timestamp"
            aggregate_interval = "12h"
            if aggregate_by == "count":
                # Group by quarter day (6-hour intervals)
                agg = df.groupby(
                    df["arrival_timestamp"].dt.floor(aggregate_interval)
                ).size()
            elif aggregate_by in ["execution_duration_ms"]:
                assert aggregate_by in df.columns, (
                    f"Column '{aggregate_by}' not found in data for series '{series_name}'.\n{df.columns}"
                )
                agg = df.groupby(df[aggregate_col].dt.floor(aggregate_interval))[
                    aggregate_by
                ].sum()
            else:
                assert aggregate_by in df.columns, (
                    f"Column '{aggregate_by}' not found in data for series '{series_name}'.\n{df.columns}"
                )
                agg = df.groupby(df[aggregate_col].dt.floor(aggregate_interval))[
                    aggregate_by
                ].mean()

            # Ensure all intervals in the aggregate range are represented, filling missing intervals with 0
            agg = agg.reindex(
                pd.date_range(
                    global_min_date, global_max_date, freq=aggregate_interval
                ),
                fill_value=0,
            )

            color = color_map[series_name]

            label_lookup = {
                "redset": "Redset",
                "generation": "Generation",
                "matching": "Matching",
                "baseline_round_robin": "Baseline (Round Robin)",
            }
            if series_name == "redset":
                # add x indicator to data points
                ax.plot(
                    agg.index,
                    agg.values,
                    label=label_lookup.get(series_name, series_name),
                    linewidth=3,
                    color=color,
                    zorder=0,
                    linestyle=":",  # Dotted line
                )
            else:
                ax.plot(
                    agg.index,
                    agg.values,
                    label=label_lookup.get(series_name, series_name),
                    alpha=0.7,
                    color=color,
                )

        if aggregate_by == "count":
            ylabel = "Number of Queries"
        elif aggregate_by in ["execution_duration_ms"]:
            ylabel = f"Runtime (ms) for interval {aggregate_interval}"
        else:
            ylabel = f"Mean of {aggregate_by.capitalize()}"

        ax.set_title(f"Cluster {cluster_id}", fontweight="bold")
        ax.set_xlabel("Arrival Time")

        if idx % n_cols == 0:
            ax.set_ylabel(ylabel)

        # first grid in last row
        if idx % n_cols == 0 and idx // n_cols == n_rows - 1:
            ax.legend(frameon=False, ncol=3, bbox_to_anchor=(3, -0.35), fontsize=14)

        ax.grid(True)

        # Set x-axis limits and ticks to show only days
        ax.set_xlim(global_min_date, global_max_date)
        xticks = pd.date_range(global_min_date, global_max_date, freq="D")
        if len(xticks) > 10:
            xticks = xticks[:: max(1, len(xticks) // 10)]
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y-%m-%d"))

        # Ensure labels are displayed
        for label in ax.get_xticklabels():
            label.set_visible(True)

        # Show x-tick labels only in the last row
        row = idx // n_cols
        if row != n_rows - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])  # Reduce left/right margins
    os.makedirs("output", exist_ok=True)
    plt.savefig(
        f"output/cluster_overview_{aggregate_by}{log_str}.png",
        dpi=300,
        bbox_inches="tight",
    )


def _get_scanset_from_redbench(row):
    # this is a serialized tuple
    if "join_tables" in row:
        jt_tuple_str = row["join_tables"]
    else:
        jt_tuple_str = row["read_tables"]

    try:
        return tuple(jt_tuple_str.split(","))
    except Exception as e:
        print(row, flush=True)
        raise e


def _compute_repetition_ratio(
    df,
    is_dml_aware: bool = False,
    is_redset: bool = False,
    is_baseline: bool = False,
    repetition_type="query_repetition",
    stop_after_1k: bool = True,
):
    assert repetition_type in ["query_repetition", "scanset_repetition"], (
        "Invalid repetition type"
    )

    # keep track of last read and write queries
    last_select = dict()
    last_write = dict()

    # count number of repetitions and total queries
    repetitions = 0
    total = 0

    for i, row in df.iterrows():
        query_type = row["query_type"]
        if query_type in ["select", "analyze", "update", "insert", "ctas"]:
            if (
                "read_table_ids" in row
                and row["read_table_ids"] is not None
                and not pd.isna(row["read_table_ids"])
                and (
                    is_redset
                    or (
                        row["join_tables"] is not None
                        and not pd.isna(row["join_tables"])
                    )
                )
            ) or (is_baseline and query_type == "select"):
                total += 1

                scanset = (
                    get_scanset_from_redset_query(row)
                    if is_redset
                    else _get_scanset_from_redbench(row)
                )
                if repetition_type == "scanset_repetition":
                    element_identifier = scanset
                elif repetition_type == "query_repetition":
                    element_identifier = row["query_hash"] if is_redset else row["sql"]
                else:
                    raise ValueError("Invalid repetition type")

                # check if no update on any table has been performed since the query was last time seen (necessary only for dml aware)
                if element_identifier in last_select:
                    ok = True
                    if is_dml_aware:
                        for table in scanset:
                            if (
                                last_write.get(table, -1)
                                > last_select[element_identifier]
                            ):
                                ok = False
                                break
                    if ok:
                        repetitions += 1

                # update timestamp when the query was last seen
                last_select[element_identifier] = i

        if is_dml_aware and query_type in [
            "update",
            "insert",
            "delete",
            "ctas",
            "copy",
        ]:
            # update timestamps when last time any caches have been invalidated / i.e. something new written to the table
            table = int(row["write_table_ids"]) if is_redset else row["write_table"]
            assert table is not None, f"Write table is None for row {row}"
            if table is not None:
                last_write[table] = i

        if stop_after_1k and i > 1000:
            break

    return repetitions / total if total else 0.0


def _plot_query_repetition_ratios(
    artifacts_dict,
    log_str: str = "",
    plot_width: int = None,
    on_first1k: bool = True,
    only_w_dml: bool = False,
):
    """
    Plot a multi-plot with query_repetition (left) and scanset_repetition (right).
    """
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(7 if plot_width is None else plot_width, 3))
    handles_labels = []
    handles_labels.append(
        _plot_query_repetition_ratios_worker(
            artifacts_dict,
            repetition_type="query_repetition",
            ax=axes[0],
            on_first1k=on_first1k,
            show_non_dml_aware=not only_w_dml,
        )
    )
    cluster_order = handles_labels[0][2]  # get cluster order from first plot
    handles_labels.append(
        _plot_query_repetition_ratios_worker(
            artifacts_dict,
            repetition_type="scanset_repetition",
            ax=axes[1],
            cluster_order=cluster_order,
            on_first1k=on_first1k,
            show_non_dml_aware=not only_w_dml,
        )
    )
    # Collect handles and labels from the first subplot only
    handles, labels, _ = handles_labels[0]

    # do not show dml
    handles = [h for h, l in zip(handles, labels) if "DML" not in l]
    labels = [l for l in labels if "DML" not in l]

    if not only_w_dml:
        # add pseudo entry explaining ... means non dml aware
        handles.append(plt.Line2D([0], [0], color="black", linestyle=":"))
        labels.append("w/o DML awareness")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.53, -0.13),
        ncol=3,
        frameon=False,
    )

    suffix = ""
    if only_w_dml:
        suffix += "_onlydml"
    if on_first1k:
        suffix += "_onfirst1k"

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs("output", exist_ok=True)
    plt.savefig(
        f"output/cluster_overview_repetition_multiplot{log_str}{suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        f"output/cluster_overview_repetition_multiplot{log_str}{suffix}.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def _plot_query_repetition_ratios_worker(
    artifacts_dict,
    repetition_type="query_repetition",
    ax=None,
    cluster_order: List[str] = None,
    on_first1k: bool = True,
    show_non_dml_aware: bool = True,
):
    # Aggregate query types for 'redset' across all clusters
    repetition_rate_dict = defaultdict(dict)
    repetition_rate_dict_wdml = defaultdict(dict)

    for cluster_id, db_dict in artifacts_dict.items():
        for db_id, series_dict in db_dict.items():
            for series_name, series_data in series_dict.items():
                df = series_data[1]

                if show_non_dml_aware:
                    repetition_rate_dict[(int(cluster_id), int(db_id))][series_name] = (
                        _compute_repetition_ratio(
                            df,
                            is_dml_aware=False,
                            is_redset=(series_name == "redset"),
                            is_baseline=("baseline" in series_name),
                            repetition_type=repetition_type,
                            stop_after_1k=on_first1k,
                        )
                    )

                repetition_rate_dict_wdml[(int(cluster_id), int(db_id))][
                    series_name
                ] = _compute_repetition_ratio(
                    df,
                    is_dml_aware=True,
                    is_redset=(series_name == "redset"),
                    is_baseline=("baseline" in series_name),
                    repetition_type=repetition_type,
                    stop_after_1k=on_first1k,
                )

    # Determine cluster order based on the 'generation' approach in descending order
    if cluster_order is None:
        cluster_order = sorted(
            repetition_rate_dict_wdml.keys(),
            key=lambda cluster_id: repetition_rate_dict_wdml[cluster_id].get(
                "redset", 0
            ),
            reverse=True,
        )
    assert len(cluster_order) > 0, "No clusters found for plotting."

    # assemble data to plot
    data = []
    series_names = set(
        [s for s_list in repetition_rate_dict_wdml.values() for s in s_list.keys()]
    )
    series_names = sorted(list(series_names))  # sort series names alphabetically

    # put baseline to the end
    if "baseline_round_robin" in series_names:
        series_names = [
            name for name in series_names if not name.startswith("baseline")
        ] + [name for name in series_names if name.startswith("baseline")]

    # extract data
    if show_non_dml_aware:
        for cluster_id in cluster_order:
            row = []
            for series_name in series_names:
                row.append(repetition_rate_dict[cluster_id].get(series_name, 0))
            data.append(row)

        data = pd.DataFrame(
            data, index=[f"{c[0]}/{c[1]}" for c in cluster_order], columns=series_names
        )

    # Prepare data for the second plot (with DML)
    data_wdml = []
    for cluster_id in cluster_order:
        row = []
        for series_name in series_names:
            row.append(repetition_rate_dict_wdml[cluster_id].get(series_name, 0))
        data_wdml.append(row)

    data_wdml = pd.DataFrame(
        data_wdml, index=[f"{c[0]}/{c[1]}" for c in cluster_order], columns=series_names
    )

    if repetition_type == "query_repetition":
        tmp = "qrr"
    else:
        tmp = "srr"

    print(f"Repetition rates (with DML {tmp}):")
    print(data_wdml)
    if show_non_dml_aware:
        print(f"Repetition rates (without DML {tmp}):")
        print(data)

    # Use provided axis or create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Define markers for different series
    markers = ["o", "s", "^", "D", "P", "*", "X", "H"]
    series_markers = {
        series: markers[i % len(markers)] for i, series in enumerate(series_names)
    }

    # Plot the data (solid for read-only, dashed for with DML)
    series_names_lookup = {
        "generation": "Redbench (Generation)",
        "matching": "Redbench (Matching)",
        "baseline_round_robin": "Baseline (Round Robin)",
        "redset": "Redset",
    }

    color_dict = {
        "generation": "#1f77b4",  # blue
        "matching": "#ff7f0e",  # orange
        "matching (join)": "#1f140b",  # dark orange
        "redset": "#fe4b4b",  # red
        "baseline_round_robin": "#2ca02c",  # green
    }

    lines = []
    labels = []
    for series_name in series_names:
        if show_non_dml_aware:
            # Read-only (dashed)
            (line,) = ax.plot(
                data.index,
                data[series_name],
                marker=series_markers[series_name] if not show_non_dml_aware else None,
                linestyle=":",
                label=f"{series_names_lookup.get(series_name, series_name)} (no DML)",
                color=color_dict.get(series_name, None),
                alpha=0.7,
                linewidth=1.5,
            )
            lines.append(line)
            labels.append(
                f"{series_names_lookup.get(series_name, series_name)} (no DML)"
            )

        # With DML (solid)
        (line,) = ax.plot(
            data_wdml.index,
            data_wdml[series_name],
            marker=series_markers[series_name] if not show_non_dml_aware else None,
            linestyle="-",
            label=f"{series_names_lookup.get(series_name, series_name)}",
            color=color_dict.get(series_name, None),
        )
        lines.append(line)
        labels.append(f"{series_names_lookup.get(series_name, series_name)}")

    ax.set_xlabel("Cluster/DB ID")

    ax.grid(True, alpha=0.7)
    ax.set_xticks(range(len(data_wdml.index)))

    ax.set_xticklabels(data_wdml.index, rotation=90, fontname="DejaVu Serif")
    ax.set_ylim(0, 1)

    # Title for each subplot
    if repetition_type == "query_repetition":
        ax.set_title("Query Repetition Rate", fontweight="bold")
        ax.set_ylabel("Query Repetition Rate")
    elif repetition_type == "scanset_repetition":
        ax.set_title("Scanset Repetition Rate", fontweight="bold")
        ax.set_ylabel("Scanset Repetition Rate")
    else:
        ax.set_title(repetition_type)

    # Do not show legend here; return handles and labels for the main legend
    return lines, labels, cluster_order
