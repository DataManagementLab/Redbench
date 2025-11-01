import json
import os
from types import SimpleNamespace

import duckdb
from matching.gen_queries.query_generator import QueryGenerator
from utils.load_and_preprocess_redset import determine_redset_dataset_type

from redbench.matching.benchmarks.imdb import IMDbBenchmark
from redbench.matching.benchmarks.tpcds import TPCDSBenchmark
from redbench.utils.log import log


def get_benchmarks_from_config(config):
    """
    Extract benchmarks configuration from the provided config.
    """
    benchmarks = []
    for benchmark_config in config.support_benchmarks:
        if benchmark_config.id.lower() == "imdb":
            benchmarks.append(IMDbBenchmark(config, benchmark_config))
        elif benchmark_config.id.lower() == "tpcds":
            benchmarks.append(TPCDSBenchmark(config, benchmark_config))
        else:
            raise ValueError(f"Unsupported benchmark ID: {benchmark_config.id}")
    return benchmarks


def generate_workload(
    output_dir,
    redset_path,
    cluster_id: int,
    database_id: int,
    config_path=None,
    overwrite_existing: bool = False,
):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "fast.json") # for fast evaluation - change to default.json for full generation

    with open(config_path, "r") as f:
        config = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    config.output_dir = os.path.join(output_dir, "tmp_matching")
    os.makedirs(config.output_dir, exist_ok=True)
    config.wl_dir = os.path.join(output_dir, "generated_workloads")
    os.makedirs(config.wl_dir, exist_ok=True)

    stats_db_filepath = os.path.join(config.output_dir, "stats.duckdb")
    # config.stats_db = duckdb.connect(stats_db_filepath, read_only=True)
    config.stats_db_filepath = stats_db_filepath

    config.redset_path = redset_path
    config.cluster_id = cluster_id
    config.database_id = database_id
    config.redset_dataset = determine_redset_dataset_type(redset_path)

    for benchmark in get_benchmarks_from_config(config):
        log("Setup DB")
        benchmark.setup_db()
        log("Setup Support Benchmark")
        benchmark.setup()
        log("Compute Stats")
        benchmark.compute_stats()
        log("Dump Plots")
        benchmark.dump_plots()

        generator = QueryGenerator(
            benchmark, config, overwrite_existing=overwrite_existing
        )
        if (
            os.path.exists(generator.workloads_dir)
            and os.path.exists(os.path.join(generator.workloads_dir, "workload.csv"))
            and not overwrite_existing
        ):
            log(
                f"Workload already exists at: {generator.workloads_dir}. Skipping creation.",
                log_mode="warning",
            )
            return

        log("Generate")
        generator.generate()
        generator.create_plots()
