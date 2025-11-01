import collections
import concurrent.futures
import os
from datetime import timedelta
from typing import Dict

import duckdb
import numpy as np
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
from utils.load_and_preprocess_redset import load_and_preprocess_redset

from redbench.generation.helper.create_join import sample_acyclic_join
from redbench.generation.helper.redset_table_sizes import (
    define_sizes_for_redset_tables,
)
from redbench.generation.helper.table_mapper import (
    map_redset_table_to_physical_table,
)
from redbench.generation.helper.tool import toggle_suffix
from redbench.generation.helper.workload_statistics_retriever import (
    DatabaseStatisticsRetriever,
)
from redbench.generation.query_builder.query_builder import (
    build_copy_query,
    build_ctas_query,
    build_delete_query,
    build_insert_select_query,
    build_select_query,
    build_update_query,
)
from redbench.utils.log import log

# Global variable to store the target database connection for workers
_worker_target_db_con = None


def _init_worker(db_augmented_path, validate_query_produces_rows):
    """
    Initializer function for process pool workers.
    Sets up a DuckDB connection for each worker process.
    """
    global _worker_target_db_con
    if validate_query_produces_rows and db_augmented_path is not None:
        _worker_target_db_con = duckdb.connect(
            database=db_augmented_path, read_only=True
        )
    else:
        _worker_target_db_con = None


def get_schema_size_for_cluster(
    con: duckdb.DuckDBPyConnection, cluster_id: int, database_id: int
) -> int:
    sql = f"""
WITH exploded AS (
            SELECT instance_id, database_id, read_id, write_id
            FROM redset_preprocessed
            LEFT JOIN UNNEST(STRING_SPLIT(read_table_ids, ',')) AS read_id ON TRUE
            LEFT JOIN UNNEST(STRING_SPLIT(write_table_ids, ',')) AS write_id ON TRUE
            WHERE query_type IN ('select', 'insert', 'update', 'delete') AND instance_id = {cluster_id} AND database_id = {database_id}
        ),
        grouped AS (
            SELECT instance_id, database_id,
                ARRAY_AGG(DISTINCT read_id) AS all_read_table_ids,
                ARRAY_AGG(DISTINCT write_id) AS all_write_table_ids,
                COUNT(DISTINCT read_id) AS num_read,
                COUNT(DISTINCT write_id) AS num_write
            FROM exploded
            GROUP BY instance_id, database_id
        ),
        all_clusters AS (
            SELECT DISTINCT instance_id, database_id FROM redset_preprocessed
        )
        SELECT ac.instance_id, ac.database_id, 
            g.all_read_table_ids, g.all_write_table_ids,
            CASE 
                WHEN g.all_read_table_ids IS NULL AND g.all_write_table_ids IS NULL THEN NULL
                ELSE g.num_read + g.num_write
            END AS schema_size
        FROM all_clusters ac LEFT JOIN grouped g ON ac.instance_id = g.instance_id AND ac.database_id = g.database_id
        ORDER BY ac.instance_id, ac.database_id
        """
    df = con.execute(sql).fetchdf()
    if len(df)!=1:
        raise Exception(
            f"No data found for cluster_id {cluster_id} and database_id {database_id}. Have you selected an cluster/database without any read/write queries? Or a timeframe where no queries were executed?"
        )
    return df.iloc[0]["schema_size"] if not pd.isna(df.iloc[0]["schema_size"]) else None


def create_workload(
    config: Dict[str, any],
    redset_path: str,
    column_statistics_path: str,
    workload_path: str,
    json_schema_path: str,
    sql_schema_path: str,
    force: bool = False,
    db_augmented_path: str = None,
):
    # create experiment directory if it does not exist
    os.makedirs(os.path.dirname(workload_path), exist_ok=True)

    interpret_deviating_mbytes_as_structural_repetition = config[
        "interpret_deviating_mbytes_as_structural_repetition"
    ]

    if (
        not force
        and os.path.exists(workload_path)
        and os.path.exists(os.path.join(workload_path, "workload.csv"))
    ):
        log(
            f"Workload already exists at: {workload_path}. Skipping creation.",
            log_mode="warning",
        )
        return

    # Define grouping columns
    group_cols = [
        "query_type",
        "num_joins",
        "num_aggregations",
        "read_table_ids",
        "write_table_ids",
        "feature_fingerprint",
        "database_id",
        "instance_id",
    ]
    group_cols_str = ", ".join(group_cols)

    # Sample size
    apply_sampling = config["apply_sampling"]
    if apply_sampling:
        sample_size = config["sample_size"]
    else:
        sample_size = 1000000000  # set to 1bn - this should be enough

    # load the redset and preprocess it
    con = load_and_preprocess_redset(
        start_date=config["start_date"],
        end_date=config["end_date"],
        redset_path=redset_path,
        instance_id=config["cluster_id"]
        if not config["enable_random_databases"]
        else None,
        database_id=config["database_id"]
        if not config["enable_random_databases"]
        else None,
        include_copy=config.get("include_copy", False),
        include_analyze=config.get("include_analyze", False),
        include_ctas=config.get("include_ctas", False),
        exclude_tables_never_read=config.get("redset_exclude_tables_never_read", False),
        limit_rows=config.get("limit_redset_rows_read", None),
    )

    randstate = np.random.RandomState(config["seed"])

    # DuckDB query for weighted sampling
    query = f"""
    WITH grouped AS (
        SELECT {group_cols_str}, COUNT(*) AS count, MAX(mbytes_scanned) AS max_mbytes_scanned
        FROM redset_preprocessed
        GROUP BY {group_cols_str}
        HAVING COUNT(*) <= {config["max_size_qig"]} 
    ),
    probabilities AS (
        SELECT *, pow(count,{config["repetition_exponent"]}) * 1.0 / SUM(count) OVER () AS probability
        FROM grouped
    ),
    ranked AS (
        SELECT *,
            (ABS(HASH({group_cols_str}, {config["seed"]})) * 1.0 / 9223372036854775807) / probability AS random_weight
        FROM probabilities
    ),
    sampled AS (
        SELECT {group_cols_str}, max_mbytes_scanned, count
        FROM ranked
        ORDER BY random_weight
        LIMIT {sample_size}
    )
    SELECT * FROM sampled
    ORDER BY {group_cols_str}
    """
    sampled_groups = con.query(query).df()

    # set max_mbytes_scanned to 1 if it is 0 - we assume 0 was produced because of rounding
    sampled_groups["max_mbytes_scanned"] = sampled_groups["max_mbytes_scanned"].replace(
        0, 0.5
    )

    # for safety reasons to ensure the original table is not used, drop it
    con.execute("DROP TABLE redset;")

    log_statistics(
        group_cols_str=group_cols_str, con=con, sampled_groups=sampled_groups
    )

    database_knowledge = DatabaseStatisticsRetriever(
        2, column_statistics_path, json_schema_path, sql_schema_path
    )
    # this maps every encountered redset table to a physical table. In maps "big" tables with the largest physical tables
    database_knowledge.compute_mapping(sampled_groups)
    tables_max_size, sampled_groups = apply_mappings_and_get_tables_max_size(
        database_knowledge, sampled_groups, config, randstate
    )

    # print warning if support database is significantly smaller than original redset schema
    redset_schema_size = get_schema_size_for_cluster(
        con, config["cluster_id"], config["database_id"]
    )

    if redset_schema_size is not None:
        # run checks - in case we can't determine the schema size, skip the check
        if redset_schema_size > len(tables_max_size):
            log(
                f"Warning: The determined schema size of the original RedSet cluster (size: {redset_schema_size}) is larger than the number of tables available in the support database ({len(tables_max_size)}). This indicates that the selected combination of redset cluster and support database might not be suitable for realistic workload generation (e.g. overly pessimistic caching estimates, different read/write patterns since multiple tables mapped, ...). Proceed with caution.",
                log_mode="warning",
            )
        else:
            log(
                f"Original RedSet schema size: {redset_schema_size}, support database schema size: {len(tables_max_size)}",
            )

    else:
        log(
            "Could not determine the schema size of the original RedSet cluster. Skipping suitability check.",
            log_mode="warning",
        )

    # determine selectivity
    join_tables_with_selectivity = []
    cap_sigma_ctr = 0
    for index, row in tqdm(
        sampled_groups.iterrows(),
        desc="Determining selectivity",
        total=len(sampled_groups),
    ):
        bytes_scanned = row["max_mbytes_scanned"]

        if row["start_t"]:
            scan_total = bytes_scanned

            table_size_total = sum(tables_max_size[t] for t in row["join_tables"])
            sigma_t = (scan_total / table_size_total) if table_size_total > 0 else 1

            if sigma_t > 1:
                log(
                    f"Warning: sigma_t > 1 ({sigma_t}) because scan_total: {scan_total} > table_size_total: {table_size_total} - set to 1 \n{dict((t, tables_max_size[t]) for t in row['join_tables'])}",
                    log_mode="debug",
                )
                sigma_t = 1
                cap_sigma_ctr += 1

            selectivity_dict = {}
            for table_name in row["join_tables"]:
                selectivity_dict[table_name] = sigma_t
            join_tables_with_selectivity.append(selectivity_dict)
        else:
            join_tables_with_selectivity.append(None)

    log(
        f"Capped sigma {cap_sigma_ctr}/{len(sampled_groups)} times to 1.0 during selectivity determination",
    )

    sampled_groups["join_tables_with_selectivity"] = join_tables_with_selectivity

    max_database = sampled_groups["database_id"].max()

    def build_retrieve_query(row):
        read_part = (
            f"= '{row['read_table_ids']}'" if row["read_table_ids"] else "IS NULL"
        )
        write_part = (
            f"= '{row['write_table_ids']}'" if row["write_table_ids"] else "IS NULL"
        )
        retrieve_group_query = f"""
            SELECT *
            FROM redset_preprocessed
            WHERE query_type = '{row["query_type"]}'
            AND num_joins = {row["num_joins"]}
            AND read_table_ids {read_part}
            AND write_table_ids {write_part}
            AND database_id = {row["database_id"]}
            AND instance_id = {row["instance_id"]}
            AND feature_fingerprint = '{row["feature_fingerprint"]}'
        """
        return retrieve_group_query

    # prepare target db connection if needed
    validate_query_produces_rows = config.get("validate_query_produces_rows", False)
    if validate_query_produces_rows:
        assert db_augmented_path is not None
        assert os.path.exists(db_augmented_path), (
            f"Augmented DB path does not exist: {db_augmented_path}"
        )
        target_db_con = duckdb.connect(database=db_augmented_path, read_only=True)
    else:
        target_db_con = None

    # parallelization strategy
    parallelize = "multiprocessing"
    # parallelize = None  # for easier debugging

    # generate tasks
    task_list = []
    final_df_list = []
    query_id = 0
    for index, row in tqdm(
        sampled_groups.iterrows(),
        desc="Assemble generation-task information",
        total=len(sampled_groups),
    ):
        group = con.query(build_retrieve_query(row)).df()

        if interpret_deviating_mbytes_as_structural_repetition:
            # sometime the mbytes scanned of repeating queries is largely different, so we generate a structural repetition of this query (same join path, but different filter literal)
            # Apply a tolerance to group similar mbytes_scanned values i.e. 50% deviation is considered still as exact repetition
            tolerance = 0.5  # 50% tolerance
            used = set()
            for idx, row_g in group.iterrows():
                if idx in used:
                    continue
                mbytes_val = row_g["mbytes_scanned"]
                similar = group[
                    (group["mbytes_scanned"] >= mbytes_val * (1 / (1 + tolerance)))
                    & (group["mbytes_scanned"] <= mbytes_val * (1 + tolerance))
                ]
                similar_idxs = set(similar.index)
                used |= similar_idxs

                # set mbytes_scanned binned
                group.loc[similar.index, "mbytes_scanned_binned"] = mbytes_val

            grouped_data = group.groupby("mbytes_scanned_binned")

            if len(grouped_data) > 1:
                log(
                    f"Different mbytes_scanned for query group {index}: {[f'{t[0]} MB ({len(t[1])} queries)' for t in grouped_data]} -> introduce structural repetition"
                )
        else:
            # take the whole group, although mbytes scanned are deviating
            grouped_data = [(-1, group)]

        for _, data in grouped_data:
            task_list.append(
                (
                    row,
                    data,
                    database_knowledge,
                    config,
                    query_id,
                    index,
                    max_database,
                    validate_query_produces_rows,
                    target_db_con if parallelize != "multiprocessing" else None,
                    tables_max_size,
                )
            )
            query_id += 1

    if parallelize is None:
        for task in tqdm(task_list, desc="Generating SQL queries"):
            repeating_query_instances = gen_query_fn(task)
            final_df_list.append(repeating_query_instances)
    elif parallelize == "multiprocessing":
        # create initializer that connects to duckdb
        max_workers = max(1, os.cpu_count())
        log(f"Using {max_workers} workers for query generation")
        with concurrent.futures.ProcessPoolExecutor(
            initializer=_init_worker,
            initargs=(db_augmented_path, validate_query_produces_rows),
            max_workers=max_workers,
        ) as executor:
            results = list(
                tqdm(
                    executor.map(gen_query_fn, task_list),
                    total=len(task_list),
                    desc="Generating SQL queries",
                )
            )
            final_df_list.extend(results)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(gen_query_fn, task_list),
                    total=len(task_list),
                    desc="Generating SQL queries",
                )
            )
            final_df_list.extend(results)

    # concatenate all dataframes in the list into a single DataFrame
    final_df = pd.concat(final_df_list, ignore_index=True)

    # close the DuckDB connections
    con.execute("DROP TABLE redset_preprocessed;")
    con.close()

    # close the target database connection if it exists
    if target_db_con is not None:
        target_db_con.close()

    final_df = final_df.sort_values(by="arrival_timestamp", ascending=True)

    final_df.to_csv(workload_path, index=False)

    log(
        f"Generated {len(sampled_groups)} (distinct) queries with in total {len(final_df)} instantiations."
    )


def gen_query_fn(task):
    global _worker_target_db_con
    (
        row,
        repeating_query_instances,
        database_knowledge,
        config,
        query_id,
        struct_id,
        max_database,
        validate_query_produces_rows,
        target_db_con,
        tables_max_size,
    ) = task
    try:
        return gen_query(
            row=row,
            repeating_query_instances=repeating_query_instances,
            database_knowledge=database_knowledge,
            config=config,
            query_id=query_id,
            struct_id=struct_id,
            max_database=max_database,
            validate_query_produces_rows=validate_query_produces_rows,
            target_db_con=_worker_target_db_con
            if _worker_target_db_con is not None
            else target_db_con,
            tables_max_size=tables_max_size,
        )
    except Exception as e:
        raise e


def relax_per_table_selectivities_by_size(
    current_join_table_selectivities: Dict[str, float],
    orig_join_table_selectivities: Dict[str, float],
    tables_max_size: Dict[str, int],
    increase_step: float = 0.1,
    portion_of_involved_tables: float = 0.2,
):
    # Sort tables by size (ascending), but also consider deviation from original selectivities
    tables = list(current_join_table_selectivities.keys())

    # Compute normalized size weights (smaller tables get higher weight)
    sizes = np.array([tables_max_size[t] for t in tables], dtype=float)
    if sizes.max() > 0:
        size_weights = 1.0 - (sizes / sizes.max())
    else:
        size_weights = np.ones_like(sizes)

    # Compute selectivity deviation weights (smaller deviation gets higher weight)
    orig_selectivities = np.array([orig_join_table_selectivities[t] for t in tables])
    curr_selectivities = np.array([current_join_table_selectivities[t] for t in tables])
    deviation = np.abs(curr_selectivities - orig_selectivities)
    if deviation.max() > 0:
        deviation_weights = 1.0 - (deviation / deviation.max())
    else:
        deviation_weights = np.ones_like(deviation)

    # Combine weights: prioritize size, but keep distribution similarity
    alpha = 0.7  # prioritize size (can be tuned)
    combined_weights = alpha * size_weights + (1 - alpha) * deviation_weights

    # Mask out tables already at selectivity 1.0
    mask = curr_selectivities < 1.0
    if not np.any(mask):
        return current_join_table_selectivities  # nothing to increase

    # num tables to increase
    num_to_increase = max(1, int(len(tables) * portion_of_involved_tables))

    # Pick the tables with the highest combined weight among those not at 1.0
    masked_weights = combined_weights * mask
    sorted_indices = np.argsort(-masked_weights)  # descending order
    tables_to_increase = [tables[i] for i in sorted_indices[:num_to_increase]]

    # Increase selectivity, but cap at 1.0
    for table_to_increase in tables_to_increase:
        current_join_table_selectivities[table_to_increase] = min(
            1.0, current_join_table_selectivities[table_to_increase] + increase_step
        )

    return current_join_table_selectivities


def gen_query(
    row: pd.Series,
    repeating_query_instances: pd.DataFrame,
    database_knowledge: DatabaseStatisticsRetriever,
    config: Dict,
    query_id: int,  # random value for seeding & identifier generation of this run
    struct_id: int,  # structural repetition id - used to identify queries with same join structure
    max_database: int,
    validate_query_produces_rows: bool = False,
    target_db_con: duckdb.DuckDBPyConnection = None,
    tables_max_size: Dict[str, int] = None,
) -> pd.DataFrame:
    def get_unique_instance(row):
        return row["instance_id"] * max_database + row["database_id"]

    randstate = np.random.RandomState(query_id)
    repeating_query_instances["start_t"] = [row["start_t"]] * len(
        repeating_query_instances
    )

    repeating_query_instances["join_tables"] = [
        ",".join(row["join_tables"]) if row["join_tables"] else None
    ] * len(repeating_query_instances)
    repeating_query_instances["write_table"] = [row["write_table"]] * len(
        repeating_query_instances
    )
    repeating_query_instances["structural_repetition_id"] = [struct_id] * len(
        repeating_query_instances
    )

    if config["enable_random_databases"]:
        repeating_query_instances["unique_db_instance"] = 0
    else:
        repeating_query_instances["unique_db_instance"] = [
            get_unique_instance(row)
        ] * len(repeating_query_instances)

    # generate the query - if validation is active and query produces no rows, try again
    attempts = 0
    query = None
    min_filter_selectivity = config["min_table_scan_selectivity"]

    # retrye configuration
    max_iterations = 1000
    strategy = "weighted_w_table_size"

    if strategy == "increase_min_sel":
        # increase min filter selectivity in case of failures to produce valid queries
        relax_every_nth_iteration = 50
        increase_amount = 0.05
    elif strategy == "weighted_w_table_size":
        # do the weighted functions
        orig_join_table_selectivities = row["join_tables_with_selectivity"].copy()
        increase_amount = 0.1
        increase_portion_of_involved_tables = 0.2

        # compute num relaxations so in the end we end up with selectivities of at least in average 0.5
        num_increases = (
            1 / (increase_amount * increase_portion_of_involved_tables) / 2
        )  # divide num_increases by 2, we don't want to fully relax to 1.0. an average of at least 0.5 is fine
        relax_every_nth_iteration = int(max_iterations / num_increases)
    else:
        raise Exception(f"Unknown strategy: {strategy}")

    # collection of queries generated but failed (to avoid repeated evaluation of same query)
    failed_query_attempts = set()

    while attempts < max_iterations:
        attempts += 1
        read_part_query = None

        if row["query_type"] in ("select", "analyze", "ctas") and row["start_t"]:
            repeating_query_instances["exact_repetition_hash"] = [query_id] * len(
                repeating_query_instances
            )
            query, read_part_query, approximated_scan_selectivities = (
                build_select_query(
                    row,
                    database_knowledge,
                    randstate,
                    min_filter_selectivity=min_filter_selectivity,
                    simple_agg=True,
                )
            )
        elif row["query_type"] == "update" and row["start_t"]:
            repeating_query_instances["exact_repetition_hash"] = [query_id] * len(
                repeating_query_instances
            )
            # Check if the table has non-primary key columns for update
            table_info = database_knowledge.retrieve_table_info(row["start_t"])
            if table_info:
                non_pk_columns = [
                    col
                    for col, props in table_info.items()
                    if not props.get("pk", False)
                ]
                if non_pk_columns:
                    query, read_part_query, approximated_scan_selectivities = (
                        build_update_query(
                            row,
                            database_knowledge,
                            randstate,
                            min_filter_selectivity=min_filter_selectivity,
                        )
                    )
                else:
                    # No non-PK columns, treat as select query
                    log(
                        f"Table {row['start_t']} has no non-primary key columns, converting UPDATE to SELECT"
                    )
                    query, read_part_query, approximated_scan_selectivities = (
                        build_select_query(
                            row,
                            database_knowledge,
                            randstate,
                            min_filter_selectivity=min_filter_selectivity,
                        )
                    )
            else:
                raise ValueError(f"Table {row['start_t']} not found in schema!")

        elif row["query_type"] == "delete" and row["start_t"]:
            query, read_part_query, approximated_scan_selectivities = (
                build_delete_query(
                    row,
                    database_knowledge,
                    randstate,
                    min_filter_selectivity=min_filter_selectivity,
                )
            )

            repeating_query_instances["exact_repetition_hash"] = [query_id] * len(
                repeating_query_instances
            )

        elif row["query_type"] == "insert" and row["start_t"]:
            if config["deactivate_repeating_inserts"]:
                q_l = []
                qr_l = []
                original_join_tables_with_selectivity = row[
                    "join_tables_with_selectivity"
                ].copy()
                approximated_scan_selectivities_list = []
                for i in range(len(repeating_query_instances)):
                    attempts = 0  # we have our own retry loop here
                    iteration_min_filter_selectivity = min_filter_selectivity
                    row["join_tables_with_selectivity"] = (
                        original_join_tables_with_selectivity.copy()
                    )

                    while attempts < max_iterations:
                        attempts += 1
                        query, read_part_query, approximated_scan_selectivities = (
                            build_insert_select_query(
                                row,
                                database_knowledge,
                                randstate,
                                config["add_conflict_logic"],
                                min_filter_selectivity=iteration_min_filter_selectivity,
                            )
                        )

                        if validate_query_produces_rows:
                            if (
                                read_part_query not in failed_query_attempts
                                and validate_query(
                                    query, read_part_query, target_db_con
                                )
                            ):
                                # query is valid - exit the generation loop
                                break
                            else:
                                # reset query and try again
                                failed_query_attempts.add(read_part_query)
                                last_attempt = query
                                query = None
                                read_part_query = None

                                # increase scan-selectivities to increase chance that generation produces a valid output
                                if attempts % relax_every_nth_iteration == 0:
                                    (
                                        iteration_min_filter_selectivity,
                                        row["join_tables_with_selectivity"],
                                    ) = apply_relax(
                                        strategy,
                                        increase_amount,
                                        attempts,
                                        row["join_tables_with_selectivity"],
                                        orig_join_table_selectivities,
                                        tables_max_size,
                                        iteration_min_filter_selectivity,
                                        increase_portion_of_involved_tables,
                                    )
                        else:
                            break

                    if query is None and last_attempt is not None:
                        log(
                            f"Warning: Could not produce a valid insert query after {attempts} attempts - use last generated query (produces no rows)",
                        )
                        query = last_attempt

                    q_l.append(query)
                    qr_l.append(read_part_query)
                    approximated_scan_selectivities_list.append(
                        approximated_scan_selectivities
                    )
                repeating_query_instances["sql"] = q_l
                repeating_query_instances["sql_read"] = qr_l
                repeating_query_instances["approximated_scan_selectivities"] = (
                    approximated_scan_selectivities_list
                )
                repeating_query_instances["exact_repetition_hash"] = [
                    f"{query_id}_{i}" for i in range(len(repeating_query_instances))
                ]

                return repeating_query_instances
            else:
                query, read_part_query, approximated_scan_selectivities = (
                    build_insert_select_query(
                        row,
                        database_knowledge,
                        randstate,
                        config["add_conflict_logic"],
                        min_filter_selectivity=min_filter_selectivity,
                    )
                )

                repeating_query_instances["exact_repetition_hash"] = [query_id] * len(
                    repeating_query_instances
                )

        elif row["query_type"] == "ctas" and row["start_t"]:
            query, read_part_query, approximated_scan_selectivities = build_ctas_query(
                row,
                database_knowledge,
                randstate,
                min_filter_selectivity=min_filter_selectivity,
            )

            repeating_query_instances["exact_repetition_hash"] = [query_id] * len(
                repeating_query_instances
            )

        elif (
            row["query_type"] == "copy"
            and row["write_table"]
            and config.get("include_copy", default=False)
        ):
            query, read_part_query, approximated_scan_selectivities = build_copy_query(
                row,
                database_knowledge,
                randstate,
                max_split_id=config["num_split"] - 1,
            )
        else:
            raise Exception(
                f"Unknown query type or missing start time for row: {row} / {row['feature_fingerprint']}"
            )

        if validate_query_produces_rows:
            if read_part_query not in failed_query_attempts and validate_query(
                query, read_part_query, target_db_con
            ):
                # query is valid - exit the generation loop
                break
            else:
                # store log info
                last_attempt = query

                # reset query and try again
                failed_query_attempts.add(read_part_query)
                query = None

                # increase scan-selectivities to increase chance that generation produces a valid output
                if attempts % relax_every_nth_iteration == 0:
                    (
                        min_filter_selectivity,
                        row["join_tables_with_selectivity"],
                    ) = apply_relax(
                        strategy,
                        increase_amount,
                        attempts,
                        row["join_tables_with_selectivity"],
                        orig_join_table_selectivities,
                        tables_max_size,
                        min_filter_selectivity,
                        increase_portion_of_involved_tables,
                    )

        else:
            break

    # fall back to last generated query in case generation was not successfull. This query might not produce any rows
    if query is None and last_attempt is not None:
        query_type = row["query_type"]
        log(
            f"Warning: Could not produce a valid {query_type} query after {attempts} attempts - use last generated query (produces no rows)",
        )
        query = last_attempt

    if query:
        repeating_query_instances["sql"] = [query] * len(repeating_query_instances)
        repeating_query_instances["approximated_scan_selectivities"] = [
            approximated_scan_selectivities
        ] * len(repeating_query_instances)
        if row["query_type"] == "ctas":
            repeating_query_instances["arrival_timestamp"] = pd.to_datetime(
                repeating_query_instances["arrival_timestamp"]
            )
            repeating_query_instances["arrival_timestamp"] = repeating_query_instances[
                "arrival_timestamp"
            ] - timedelta(seconds=20)  # redset introduced noise for each query
        return repeating_query_instances
    else:
        raise Exception(
            f"Could not build query for row: {row} \n {row['join_tables_with_selectivity']}\n{last_attempt}"
        )


def apply_relax(
    strategy: str,
    increase_amount: float,
    attempts: int,
    join_tables_with_selectivity,
    orig_join_table_selectivities: Dict[str, float],
    tables_max_size: Dict[str, int],
    min_filter_selectivity: float,
    increase_portion_of_involved_tables: float,
):
    if strategy == "increase_min_sel":
        min_filter_selectivity = min(
            1.0, max(0, min_filter_selectivity) + increase_amount
        )
        log(
            f"Increase min_filter_selectivity to {min_filter_selectivity:.2f} after {attempts} failed attempts to produce a valid query",
        )
    elif strategy == "weighted_w_table_size":
        # apply weighted relaxation prioritizing smaller tables first
        join_tables_with_selectivity = relax_per_table_selectivities_by_size(
            join_tables_with_selectivity,
            orig_join_table_selectivities,
            tables_max_size,
            increase_step=increase_amount,
            portion_of_involved_tables=increase_portion_of_involved_tables,
        )
        log(
            f"Relaxed table selectivities to {join_tables_with_selectivity} after {attempts} failed attempts to produce a valid query",
            log_mode="debug",
        )
    else:
        raise Exception(f"Unknown strategy: {strategy}")

    return min_filter_selectivity, join_tables_with_selectivity


def validate_query(
    query: str, read_part_query: str, target_db_con: duckdb.DuckDBPyConnection
):
    # validate that the read_part produces at least one row
    assert read_part_query.startswith("SELECT"), (
        "read_part_query must be a SELECT statement"
    )

    # extract the FROM clause
    from_clause = read_part_query.split(" FROM ", 1)[1].strip(";")

    # replace any potential aggregation with a simple count
    exists_query = f"SELECT EXISTS (SELECT 1 FROM {from_clause});"
    try:
        exists_result = target_db_con.execute(exists_query).fetchone()
        if exists_result is None or not exists_result[0]:
            # log(f"Validation failed: read part produces no rows: {read_part_query}")
            return False
        return True
    except Exception as e:
        # if validation query fails, log and return False to try again with different parameters
        log(f"Validation query failed: {exists_query}, error: {e}", log_mode="debug")
        return False


def log_statistics(
    group_cols_str: str, con: duckdb.DuckDBPyConnection, sampled_groups: pd.DataFrame
):
    # compute total number of QIGs in the dataset
    count_all_qigs_query = f"""
    SELECT COUNT(*) AS count_per_qig, query_type
        FROM redset_preprocessed
        GROUP BY {group_cols_str}
    """
    redset_count_stats = con.query(count_all_qigs_query).df()
    # redset_num_qigs = len(redset_count_stats)

    # log number of querytypes
    wl_num_qigs_per_type = sampled_groups["query_type"].value_counts()
    wl_num_queries_per_type = sampled_groups.groupby("query_type")["count"].sum()

    redset_num_qigs_per_type = redset_count_stats["query_type"].value_counts()

    redset_num_queries_per_type = con.query(
        "SELECT query_type, COUNT(*) AS count FROM redset_preprocessed GROUP BY query_type"
    ).df()

    # convert to series
    redset_num_queries_per_type = pd.Series(
        redset_num_queries_per_type.set_index("query_type")["count"]
    )

    # Combine into a DataFrame for easier tabulation
    query_type_counts = (
        pd.DataFrame(
            {
                "wl_unique_queries": wl_num_qigs_per_type,
                "wl_num_queries": wl_num_queries_per_type,
                "redset_unique_queries": redset_num_qigs_per_type,
                "redset_num_queries": redset_num_queries_per_type,
            }
        )
        .fillna(0)
        .astype(int)
    )

    # Add a sum row
    sum_row = query_type_counts.sum(numeric_only=True)
    sum_row.name = "Total"
    query_type_counts = pd.concat([query_type_counts, sum_row.to_frame().T])

    log("Sampled Queries - Query Type Counts:")
    log(
        tabulate(
            query_type_counts.reset_index().values,
            headers=query_type_counts.columns,
            tablefmt="github",
        )
    )


def apply_mappings_and_get_tables_max_size(
    database_knowledge: DatabaseStatisticsRetriever,
    sampled_groups: pd.DataFrame,
    config: Dict,
    randstate: np.random.RandomState,
):
    relationships_table = collections.defaultdict(list)
    for (
        table_l,
        column_l,
        table_r,
        column_r,
    ) in database_knowledge.retrieve_relationships():
        if not isinstance(column_l, list):
            column_l = [column_l]
        if not isinstance(column_r, list):
            column_r = [column_r]

        relationships_table[table_l].append([column_l, table_r, column_r])
        relationships_table[table_r].append([column_r, table_l, column_l])

    # a dict of the current physical db with its tables and their sizes
    tables_max_size = {key: 0 for key in database_knowledge.get_original_table_names()}
    redset_table_sizes, _, _ = define_sizes_for_redset_tables(sampled_groups)
    redset_tables = [key for key, value in redset_table_sizes.items()]

    start_table_column = []
    joins_column = []
    join_tables_column = []
    write_table_column = []

    sampled_groups = sampled_groups.sort_values(
        by="query_type", ascending=False, key=lambda x: x == "ctas"
    )
    for index, row in tqdm(
        sampled_groups.iterrows(),
        desc="Mapping table ids to schema",
        total=len(sampled_groups),
    ):
        bytes_scanned = row["max_mbytes_scanned"]

        if bytes_scanned > 0 or row["query_type"] in ["copy"]:
            # log(row)
            chosen_table = None
            if config["enable_random_table_ids"]:
                chosen_table = randstate.choice(redset_tables)
            else:
                if row["query_type"] in ("select", "analyze"):
                    if row["read_table_ids"]:
                        r_tables = row["read_table_ids"].split(",")

                        if config["start_mapping_largest_table"]:
                            # start the mapping with the largest table - from there sample joins by random walks in the relationship graph
                            chosen_table = max(
                                r_tables,
                                key=lambda t: redset_table_sizes.get(t, -1),
                                default=None,
                            )

                            if chosen_table is None:
                                # fallback to random choice
                                chosen_table = randstate.choice(r_tables)

                        else:
                            chosen_table = randstate.choice(r_tables)
                else:
                    if row["write_table_ids"]:
                        chosen_table = row["write_table_ids"]

            if chosen_table:
                chosen_table = int(chosen_table)
                assert chosen_table in database_knowledge.retrieve_mapping(), (
                    f"Chosen table id {chosen_table} not in mapping {database_knowledge.retrieve_mapping()}\n{row['query_type']}"
                )
                if chosen_table in database_knowledge.retrieve_mapping():
                    start_t = database_knowledge.retrieve_mapping()[chosen_table]
                else:
                    # the initial mapping did not cover this table - we had no little statistics. Map it randomly
                    start_t = map_redset_table_to_physical_table(
                        chosen_table, database_knowledge.get_original_table_names()
                    )

                if row["query_type"] in ("select", "analyze", "ctas"):
                    write_table_column.append(None)
                elif row["query_type"] == "insert":
                    write_table_column.append(toggle_suffix(start_t))
                elif row["query_type"] == "ctas":
                    table_name = start_t + "_ctasc2b89z8c2z9_" + str(index)
                    database_knowledge.update_mapping(chosen_table, table_name)
                    write_table_column.append(table_name)
                    tables_max_size[table_name] = bytes_scanned

                    table_info = database_knowledge.retrieve_table_info(start_t)

                    pk_columns = [
                        col
                        for col, props in table_info.items()
                        if props.get("pk", False)
                    ]
                    relationships_table[start_t].append(
                        [pk_columns, table_name, pk_columns]
                    )
                    relationships_table[table_name].append(
                        [pk_columns, start_t, pk_columns]
                    )
                    database_knowledge.add_new_relation(
                        (start_t, pk_columns, table_name, pk_columns)
                    )
                else:
                    write_table_column.append(start_t)
            else:
                start_table_column.append(None)
                joins_column.append(None)
                join_tables_column.append(None)
                write_table_column.append(None)

                continue

            joins, join_tables = sample_acyclic_join(
                start_t,
                database_knowledge.get_original_table_names() + [start_t],
                row["num_joins"],
                relationships_table,
                database_knowledge.retrieve_table_info(),
                randstate,
                1,
            )
            start_table_column.append(start_t)
            joins_column.append(joins)
            join_tables_column.append(join_tables)

            if len(join_tables) == 1:
                tables_max_size[start_t] = max(tables_max_size[start_t], bytes_scanned)

        else:
            start_table_column.append(None)
            joins_column.append(None)
            join_tables_column.append(None)
            write_table_column.append(None)

    sampled_groups["start_t"] = start_table_column
    sampled_groups["joins_t"] = joins_column
    sampled_groups["join_tables"] = join_tables_column
    sampled_groups["write_table"] = write_table_column

    # add some slack to tables_max_size
    for t in tables_max_size:
        tables_max_size[t] = int(tables_max_size[t] * 2)

    return tables_max_size, sampled_groups
