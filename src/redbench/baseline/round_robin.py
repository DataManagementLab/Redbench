# The round-robin baseline includes the arrival times of the redset. It executes the queries of the support benchmark in a round-robin fashion.
# For the DMLs simple delete/insert statements are used (like matching-based).

import argparse
import json
import os
import random
import sys

import duckdb
import pandas
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generation.run import gen_expname_from_config
from matching.benchmarks.imdb import IMDbBenchmark
from matching.benchmarks.tpcds import TPCDSBenchmark
from matching.gen_queries.matching_utils import SimpleDmlsConstructor, init_query
from plots.create_plots import create_plots
from utils.load_and_preprocess_redset import load_and_preprocess_redset
from utils.log import log


def generate_round_robin(
    instance_id: int,
    database_id: int,
    start_date: str,
    end_date: str,
    output_dir: str,
    redset_dataset: str,
    redset_path: str,
    support_benchmark: str,
    random_arrival_time: bool = True,
    one_instance_per_template: bool = False,
    align_queries_with_num_distinct_in_redset: bool = True,
    limit_queries: int = None,
) -> str:
    # assemble baseline config
    baseline_config = {
        "support_benchmark_arg": support_benchmark,
        "start_date": start_date,
        "end_date": end_date,
        "use_table_versioning": False,
        "random_arrival_time": random_arrival_time,
        "one_instance_per_template": one_instance_per_template,
        "redset_path": redset_path,
        "cluster_id": instance_id,
        "database_id": database_id,
        "align_queries_with_num_distinct_in_redset": align_queries_with_num_distinct_in_redset,
    }
    if limit_queries is not None:
        baseline_config["limit_queries"] = limit_queries

    # get baseline config hash
    exp_name = gen_expname_from_config(baseline_config, strategy="baseline_round_robin")

    # get output dir
    baseline_out_dir = os.path.join(
        output_dir,
        "generated_workloads",
        support_benchmark,
        redset_dataset,
        f"cluster_{instance_id}",
        f"database_{database_id}",
        exp_name,
    )
    out_path = os.path.join(baseline_out_dir, "workload.csv")
    os.makedirs(baseline_out_dir, exist_ok=True)

    if os.path.exists(out_path):
        log(f"Round robin baseline already exists: {out_path}", log_mode="error")
        return baseline_out_dir

    config = {
        "output_dir": os.path.join(output_dir, "tmp_matching"),
    }

    # convert to namespace
    config = type("Config", (object,), config)

    stats_db_filepath = os.path.join(config.output_dir, "stats.duckdb")
    config.stats_db = duckdb.connect(stats_db_filepath, read_only=True)
    config.stats_db_filepath = stats_db_filepath

    if support_benchmark == "imdb":
        benchmark_config = {
            "id": "imdb",
            "name": "imdb",
            "stats_table": "imdb_stats",
            "table_ids_table": "imdb_table_ids",
            "override": False,
        }
        # convert to namespace
        benchmark_config = type("BenchmarkConfig", (object,), benchmark_config)
        benchmark = IMDbBenchmark(config, benchmark_config)
    elif support_benchmark == "tpcds":
        benchmark = TPCDSBenchmark(config, benchmark_config)
    else:
        raise ValueError(f"Unsupported benchmark: {support_benchmark}")

    query_list = []

    # get benchmark stats
    benchmark._load_table_ids()
    benchmark_stats = benchmark.get_stats()
    table_names = benchmark.get_table_names()

    # load query filepaths
    processed_templates = set()
    for query_filepath in benchmark.get_stats().keys():
        if "job/" in query_filepath:
            # always add
            pass
        elif "ceb/" in query_filepath:
            # extract template
            if one_instance_per_template:
                # check if template was already used
                template = benchmark._extract_template_from_filepath(query_filepath)
                if template in processed_templates:
                    # skip
                    continue
                processed_templates.add(template)
        else:
            raise ValueError(f"Unexpected query filepath: {query_filepath}")

        query = init_query(query_filepath, table_versions=None, remove_linebreaks=True)
        assert isinstance(query, str), f"Query is not a string: {type(query)}"

        # annotate scanset
        scanset = [
            table_names[table] for table in benchmark_stats[query_filepath]["scanset"]
        ]
        query_list.append((query, scanset))

    # randomly shuffle - use random generator seed
    rnd = random.Random(42)
    rnd.shuffle(query_list)
    log(f"Loaded {len(query_list)} benchmark queries.")

    redset_df = (
        load_and_preprocess_redset(
            start_date=start_date,
            end_date=end_date,
            redset_path=redset_path,
            database_id=database_id,
            instance_id=instance_id,
            limit_rows=limit_queries,
        )
        .execute("SELECT * FROM redset_preprocessed;")
        .df()
    )

    if align_queries_with_num_distinct_in_redset:
        # extract num unique queries in redset
        num_unique = redset_df[redset_df["query_type"] == "select"][
            "query_hash"
        ].nunique()

        # align query list length with num unique queries in redset
        if len(query_list) > num_unique:
            query_list = query_list[:num_unique]
            log(
                f"Aligning number of benchmark queries ({len(query_list)}) with number of distinct read queries in redset ({num_unique})."
            )

    # convert arrival timestamps to pandas datetime
    redset_df["arrival_timestamp"] = pandas.to_datetime(redset_df["arrival_timestamp"])

    # get r/w ratio
    num_select_queries = redset_df[redset_df["query_type"] == "select"].shape[0]
    num_total_queries = redset_df.shape[0]
    read_ratio = num_select_queries / num_total_queries

    if random_arrival_time:
        # assign queries randomly according to read ratio
        min_timestamp = redset_df["arrival_timestamp"].min()
        max_timestamp = redset_df["arrival_timestamp"].max()
        time_step = (max_timestamp - min_timestamp) / len(redset_df)

        # generate baseline queries
        baseline_rows = []

        # counters for r/w ratio enforcement and round robin
        read_ctr = 0
        write_ctr = 0
        query_list_idx = 0
        timestamp = min_timestamp

        # assemble dml constructor
        dml_constructor = SimpleDmlsConstructor(benchmark.get_db())
        table_names = list(benchmark.get_table_names().values())
        rnd = random.Random(42)

        for i in tqdm(range(len(redset_df)), desc="Generating round robin baseline"):
            # determine if read or write
            expected_reads = (i + 1) * read_ratio
            if read_ctr < expected_reads:
                # assign read query
                query, scanset = query_list[query_list_idx % len(query_list)]
                query_list_idx += 1
                read_ctr += 1
                query_type = "select"
                write_table = None
            else:
                # assign write query
                # pick random table to write to
                write_table = rnd.choice(table_names)

                # generate random dml
                query = dml_constructor({"benchmark_write_table": write_table})
                write_ctr += 1
                query_type = "insert"
                scanset = None

            baseline_rows.append(
                {
                    "arrival_timestamp": timestamp,
                    "query_type": query_type,
                    "sql": query,
                    "read_tables": ",".join(scanset) if scanset is not None else None,
                    "write_table": write_table,
                }
            )

            timestamp += time_step

        # convert to dataframe
        baseline_df = pandas.DataFrame(baseline_rows)

    else:
        # This is an unrealistic assumption for a standard-benchmark baseline (standard benchmarks do not involve information about arrival times).
        # Hence, this baseline implementation is kind of halfway between standard benchmark and redbench

        # round robin index
        query_list_idx = 0
        baseline_df = redset_df.copy()
        for idx, row in baseline_df.iterrows():
            if row["query_type"] != "select":
                continue

            # assing query in round robin format
            query, scanset = query_list[query_list_idx % len(query_list)]
            baseline_df.at[idx, "sql"] = query

            # get scanset for this query
            baseline_df.at[idx, "read_tables"] = ",".join(scanset)

            query_list_idx += 1

            # clean other entries
            cols_to_clean = [
                "join_tables",
                # "filepath",
                # "versioning",
                "exact_repetition_hash",
                "structural_repetition_id",
                "approximated_scan_selectivities",
            ]
            for col in cols_to_clean:
                assert col in baseline_df.columns, (
                    f"Column {col} not in dataframe. {baseline_df.columns}"
                )
                baseline_df = baseline_df.drop(columns=[col])

    # copy config to baseline dir
    dst_config_path = os.path.join(baseline_out_dir, "used_config.json")
    with open(dst_config_path, "w") as f:
        json.dump(baseline_config, f, indent=4)

    # write to output dir
    log(f"Writing round robin baseline to {baseline_out_dir}")
    baseline_df.to_csv(out_path, index=False)

    return baseline_out_dir


if __name__ == "__main__":
    # arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output_dir",
        type=str,
        default="output/round_robin",
        help="Output directory",
    )
    argparser.add_argument(
        "--support_benchmark",
        type=str,
        default="imdb",
        help="Support benchmark to use (imdb or tpcds)",
    )
    argparser.add_argument(
        "--redset_dataset",
        type=str,
        default="provisioned",
        help="Redset dataset name (used for output dir structure)",
    )
    argparser.add_argument(
        "--redset_path",
        type=str,
        required=True,
        help="Redset dataset path (used for output dir structure)",
    )
    argparser.add_argument(
        "--instance_id",
        type=int,
        required=True,
        help="Instance ID to use",
    )
    argparser.add_argument(
        "--database_id",
        type=int,
        required=True,
        help="Database ID to use",
    )
    argparser.add_argument(
        "--start_date",
        type=str,
        help="Start date for redset filtering (YYYY-MM-DD)",
    )
    argparser.add_argument(
        "--end_date",
        type=str,
        help="End date for redset filtering (YYYY-MM-DD)",
    )
    argparser.add_argument(
        "--limit_queries",
        type=int,
        default=None,
        help="Limit the number of queries (default: None)",
    )

    args = argparser.parse_args()

    baseline_out_dir = generate_round_robin(
        instance_id=args.instance_id,
        database_id=args.database_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        redset_dataset=args.redset_dataset,
        redset_path=args.redset_path,
        support_benchmark=args.support_benchmark,
        limit_queries=args.limit_queries,
    )

    # create plots
    config = json.load(open(os.path.join(baseline_out_dir, "used_config.json"), "r"))
    log("Creating plots for round robin baseline...")
    create_plots(
        baseline_out_dir,
        None,
        config["redset_path"],
        gen_strategy="baseline_round_robin",
    )

    log("done")
