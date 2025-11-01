import ast
import os
import re

import pandas as pd
from generation.query_builder.gen_wl_weighted_sampling import load_and_preprocess_redset
from plots.plot_arrival_time_overview import (
    plot_aggregated_arrival_times,
    plot_arrival_time_by_query_type,
)
from plots.plot_query_type_frequencies import plot_query_type_frequencies
from plots.plot_read_write_timeline import (
    plot_read_write_timeline,
)
from plots.plot_repetition import plot_repetition
from plots.plot_scanned_rows import plot_scanned_rows
from utils.log import log


def create_plots(
    exp_dir_path, column_stats_path: str, redset_dataset_path: str, gen_strategy: str
):
    # load generated workload
    gen_wl_path = os.path.join(exp_dir_path, "workload.csv")
    config_path = os.path.join(exp_dir_path, "used_config.json")
    plot_dir = os.path.join(exp_dir_path, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # load config and extract parameters
    import json

    with open(config_path, "r") as f:
        config = json.load(f)

    if column_stats_path is not None:
        with open(column_stats_path, "r") as f:
            column_stats = json.load(f)
    else:
        column_stats = None

    # database_name = config['database_name']
    database_id = config["database_id"]
    instance_id = config["cluster_id"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    limit_redset_rows_read = config.get("limit_redset_rows_read", None)

    # load generated workload
    wl_df = pd.read_csv(
        gen_wl_path, parse_dates=["arrival_timestamp"], low_memory=False
    )

    con = load_and_preprocess_redset(
        start_date,
        end_date,
        database_id=database_id,
        instance_id=instance_id,
        redset_path=redset_dataset_path,
        include_copy=config.get("include_copy", False),
        include_analyze=config.get("include_analyze", False),
        include_ctas=config.get("include_ctas", False),
        exclude_tables_never_read=config.get("redset_exclude_tables_never_read", False),
        limit_rows=limit_redset_rows_read,
    )

    # retrieve queries from redset
    redset_entries = con.execute("SELECT * FROM redset_preprocessed;").df()

    # Ensure arrival_timestamp is datetime
    def ensure_datetime(df):
        if not pd.api.types.is_datetime64_any_dtype(df["arrival_timestamp"]):
            df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
        return df

    redset_entries = ensure_datetime(redset_entries)
    wl_df = ensure_datetime(wl_df)

    # annotate number of rows scanned
    if column_stats is not None:
        wl_df = _annotate_num_rows_scanned(wl_df, column_stats)

    # plot absolute and relative occurences of the different query types (select, insert, etc.)
    plot_query_type_frequencies(
        redset_entries=redset_entries,
        wl_df=wl_df,
        artifacts_dir=plot_dir,
        gen_strategy=gen_strategy,
    )

    plot_repetition(redset_entries=redset_entries, wl_df=wl_df, artifacts_dir=plot_dir)

    # plot_query_runs(output_workload_path, os.path.join(plots_path, 'query_runs.png'))

    # # compute a scale factor - since we have sampled down the redset, we need to scale the workload
    # num_queries_scale_factor = len(redset_entries) / len(wl_df) # do not apply scale factor - would have to be displayed on separate axis. Confuses the reader

    plot_aggregated_arrival_times(
        aggregate_by="count",
        redset_entries=redset_entries,
        wl_df=wl_df,
        num_queries_scale_factor=1,
        artifacts_dir=plot_dir,
        gen_strategy=gen_strategy,
    )
    try:
        plot_aggregated_arrival_times(
            aggregate_by="num_joins",
            redset_entries=redset_entries,
            wl_df=wl_df,
            num_queries_scale_factor=1,
            artifacts_dir=plot_dir,
            gen_strategy=gen_strategy,
        )
    except Exception as e:
        log(f"Could not plot aggregated arrival times by num_joins: {e}")

    try:
        plot_aggregated_arrival_times(
            aggregate_by="num_scans",
            redset_entries=redset_entries,
            wl_df=wl_df,
            num_queries_scale_factor=1,
            artifacts_dir=plot_dir,
            gen_strategy=gen_strategy,
        )
    except Exception as e:
        log(f"Could not plot aggregated arrival times by num_scans: {e}")

    try:
        plot_arrival_time_by_query_type(
            redset_entries=redset_entries,
            wl_df=wl_df,
            scale_factor=1,
            artifacts_dir=plot_dir,
            gen_strategy=gen_strategy,
        )
    except Exception as e:
        log(f"Could not plot arrival time by query type: {e}")

    try:
        artifacts_dict = {instance_id: {database_id: {"gen_strategy": (None, wl_df)}}}
        plot_read_write_timeline(
            artifacts_dict=artifacts_dict,
            plot_dir=plot_dir,
            cluster_id=instance_id,
            db_id=database_id,
        )
    except Exception as e:
        log(f"Could not plot read write timeline: {e}")

    if column_stats is not None:
        try:
            plot_scanned_rows(
                redset_entries=redset_entries,
                wl_df=wl_df,
                artifacts_dir=plot_dir,
                query_type="all",
            )
        except Exception as e:
            log(f"Could not plot scanned rows for all queries: {e}")


def _annotate_num_rows_scanned(wl_df: pd.DataFrame, column_stats: dict):
    # augment wl_df with approximated rows scanned per query
    # initialize table rows from column statistics
    table_rows_dict = {}
    for table, stats in column_stats.items():
        table_rows_dict[table] = stats["total_rows"]

    def clean_table(table_name: str) -> str:
        # remove the _<number> suffix from the table name if it exists
        return re.sub(r"_\d+$", "", table_name)

    def get_rows_scanned(query_row):
        # extract tables
        tables = query_row["join_tables"]
        approximated_scan_selectivities = query_row["approximated_scan_selectivities"]

        # convert string to list of tuples
        if isinstance(approximated_scan_selectivities, str):
            approximated_scan_selectivities = ast.literal_eval(
                approximated_scan_selectivities
            )

        # convert tables to list if it's a string
        if isinstance(tables, str):
            tables = tables.split(",")
            tables = [
                t.strip() for t in tables if t.strip()
            ]  # remove empty strings and strip whitespace

        assert isinstance(approximated_scan_selectivities, list) or isinstance(
            approximated_scan_selectivities, tuple
        ), (
            f"approximated_scan_selectivities must be a list or tuple, got {type(approximated_scan_selectivities)} / {approximated_scan_selectivities}"
        )

        assert len(tables) >= len(approximated_scan_selectivities), (
            f"Number of tables ({len(tables)}) must be greater than or equal to number of approximated scan selectivities ({len(approximated_scan_selectivities)})"
        )

        # extract rows scanned from approximated scan selectivities
        scanned_rows = 0

        # iterate over approximated scan selectivities
        processed_tables = set()
        for table, column, selectivity in approximated_scan_selectivities:
            assert table not in processed_tables, (
                f"duplicate table in approximated scan selectivities: {table}"
            )
            processed_tables.add(table)

            cleaned_table = clean_table(table)  # clean table name
            scanned_rows += int(table_rows_dict[cleaned_table] * selectivity)

        for table in tables:
            if table not in processed_tables:
                # if table is not in approximated scan selectivities, use total rows
                cleaned_table = clean_table(table)
                scanned_rows += table_rows_dict[cleaned_table]

        return scanned_rows

    # sort wl_df by arrival_timestamp
    wl_df = wl_df.sort_values(by="arrival_timestamp")

    # iterate over workload and add rows scanned
    rows_scanned = []
    for _, row in wl_df.iterrows():
        rows_scanned.append(get_rows_scanned(row))
    wl_df["rows_scanned"] = rows_scanned

    return wl_df
