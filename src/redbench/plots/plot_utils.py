import json
import os
from typing import Dict

import duckdb
import pandas as pd
from generation.query_builder.gen_wl_weighted_sampling import load_and_preprocess_redset


def _ensure_datetime(df):
    if not pd.api.types.is_datetime64_any_dtype(df["arrival_timestamp"]):
        df["arrival_timestamp"] = pd.to_datetime(df["arrival_timestamp"])
    return df


def _load_wl_artifacts(base_wl_path: str, exp_path: str):
    # Load the workload artifacts from the specified paths
    config_path = os.path.join(base_wl_path, exp_path, "used_config.json")
    workload = os.path.join(base_wl_path, exp_path, "workload.csv")

    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    assert os.path.exists(workload), f"Workload file not found: {workload}"

    with open(config_path, "r") as f:
        config = json.load(f)

    # load generated workload
    wl_df = pd.read_csv(workload, parse_dates=["arrival_timestamp"], low_memory=False)
    wl_df = _ensure_datetime(wl_df)

    return config, wl_df


def _check_config_similarity(artifacts_dict):
    # check that all configs have the same database_id, cluster_id, start_date, and end_date
    keys_to_check = ["database_id", "cluster_id", "start_date", "end_date"]
    reference = None

    for exp_name, (config, _) in artifacts_dict.items():
        current = dict((k, config.get(k)) for k in keys_to_check)
        if reference is None:
            reference = current
        else:
            if current != reference:
                raise ValueError(
                    f"Config mismatch in experiment '{exp_name}':\n"
                    f"Expected {dict((k, reference[k]) for k in keys_to_check)},\n"
                    f"but got {dict((k, current[k]) for k in keys_to_check)}"
                )

    assert reference is not None, "No valid experiments found."


def add_redset_entries(
    artifacts_dict: Dict,
    redset_path: str,
    exclude_tables_never_read: bool = False,
    reference_config: Dict = None,
    load_full_redset_and_cache: bool = False,
    raise_assertion_if_no_experiments_found: bool = True,  # can be deactivated in case we only look for redset entries
    duckdb_con: duckdb.DuckDBPyConnection = None,
):
    for cluster_id, db_dict in artifacts_dict.items():
        for db_id, exp_dict in db_dict.items():
            if raise_assertion_if_no_experiments_found:
                assert len(exp_dict) > 0, (
                    f"No experiments found for cluster {cluster_id}, db {db_id}"
                )

            if reference_config is None:
                # make sure all configs match
                # _check_config_similarity(exp_dict)

                # get the first one, all have the same entries as checked before
                reference_config = exp_dict[next(iter(exp_dict))][0]

                # extract the common parameters
                assert reference_config["database_id"] == db_id, (
                    f"Expected database_id {db_id}, got {reference_config['database_id']}"
                )
                assert reference_config["cluster_id"] == cluster_id, (
                    f"Expected cluster_id {cluster_id}, got {reference_config['cluster_id']}"
                )

            # extract start_date, end_date
            start_date = reference_config["start_date"]
            end_date = reference_config["end_date"]
            limit_readset_rows = reference_config.get("limit_redset_rows_read", None)

            # load the redset
            duckdb_con_loaded = load_and_preprocess_redset(
                start_date,
                end_date,
                database_id=db_id,
                instance_id=cluster_id,
                redset_path=redset_path,
                include_copy=reference_config.get("include_copy", False),
                include_analyze=reference_config.get("include_analyze", False),
                include_ctas=reference_config.get("include_ctas", False),
                exclude_tables_never_read=exclude_tables_never_read,
                load_full_redset_and_cache=load_full_redset_and_cache,
                con=duckdb_con,
                limit_rows=limit_readset_rows,
            )
            redset_entries = duckdb_con_loaded.execute(
                "SELECT * FROM redset_preprocessed;"
            ).df()

            print(f"Num entries retrieved from redset: {len(redset_entries)}")

            # Ensure arrival_timestamp is datetime
            redset_entries = _ensure_datetime(redset_entries)

            # add redset entries to the artifacts
            artifacts_dict[cluster_id][db_id]["redset"] = (
                None,
                redset_entries,
            )  # none = config value is expected on first position

    return duckdb_con_loaded
