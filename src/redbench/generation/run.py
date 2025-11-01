import hashlib
import json
import os
from typing import Dict

from download_artifacts import download_artifacts
from utils.load_and_preprocess_redset import determine_redset_dataset_type

from redbench.generation.dataset_input.create_db import create_duckdb
from redbench.generation.dataset_input.create_normalized_datasets import (
    create_normalized_dataset,
)
from redbench.generation.dataset_input.prepare_and_scale import (
    prepare_and_scale_dataset,
)
from redbench.generation.dataset_input.retrieve_statistics import create_quantiles
from redbench.generation.query_builder.gen_wl_weighted_sampling import (
    create_workload,
)
from redbench.plots.create_plots import create_plots
from redbench.utils.log import log


def gen_expname_from_config(config: Dict, strategy: str) -> str:
    # remove the cluster-id and database id from config (this is captured in the directories and should not change the workload hash)

    tmp_config = config.copy()

    if "cluster_id" in tmp_config:
        tmp_config.pop("cluster_id")
    if "database_id" in tmp_config:
        tmp_config.pop("database_id")
    if "redset_path" in tmp_config:
        tmp_config.pop("redset_path")

    if strategy in ["matching", "generation"]:
        assert "redset_dataset" in tmp_config, (
            f"redset_dataset must be specified in the config: {tmp_config}"
        )
    if "redset_dataset" in tmp_config:
        tmp_config.pop("redset_dataset")

    experiment_name = (
        strategy
        + "_"
        + hashlib.md5(json.dumps(tmp_config, sort_keys=True).encode()).hexdigest()
    )

    return experiment_name


def generate_workload(
    output_path: str,
    redset_path: str,
    instance_id: int,
    database_id: int,
    config_path: str = None,
    force_workload_creation: bool = False,
):
    # Implementation of the workload generation logic

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "fast.json") # for fast evaluation - change to default.json for full generation

    with open(config_path, "r") as f:
        config = json.load(f)

    database_name = config["database_name"]
    if database_name is None:
        raise ValueError("Database name must be specified in the config.")
    if database_name in ["imdb", "baseball"]:
        # check if example database schema & data has been downloaded already
        if not os.path.exists(
            os.path.join(output_path, "tmp_generation", database_name, 'schema.json')
        ):
            download_artifacts(artifacts_dir=output_path, databases=[database_name])

    # Add instance_id and database_id to the config
    config["cluster_id"] = instance_id
    config["database_id"] = database_id
    config["redset_path"] = redset_path
    config["redset_dataset"] = determine_redset_dataset_type(redset_path)

    experiment_name = gen_expname_from_config(config, strategy="generation")

    # Extract parameters
    csv_directory_path = config["raw_database_tables"]
    db_name = config["database_name"]
    augmentation_factor = config["schema_augmentation_factor"]

    # assemble output path
    generation_tmp_path = os.path.join(output_path, "tmp_generation")
    os.makedirs(os.path.join(generation_tmp_path, db_name), exist_ok=True)

    normalized_tables_path = os.path.join(
        generation_tmp_path, db_name, "tables_normalized"
    )
    split_csv_directory_path = os.path.join(
        generation_tmp_path, db_name, f"tables_augmented_x{augmentation_factor}"
    )

    statistics_path = os.path.join(
        generation_tmp_path, db_name, "column_statistics.json"
    )
    output_original_db_path = os.path.join(
        generation_tmp_path, db_name, "db_original.duckdb"
    )
    output_augmented_db_path = os.path.join(
        generation_tmp_path, db_name, f"db_augmented_x{augmentation_factor}.duckdb"
    )
    output_schema_path = os.path.join(
        generation_tmp_path, db_name, f"schema_augmented_x{augmentation_factor}.sql"
    )
    output_workload_path = os.path.join(
        generation_tmp_path, db_name, experiment_name, "workload.csv"
    )
    output_config_path = os.path.join(
        generation_tmp_path, db_name, experiment_name, "used_config.json"
    )
    json_schema_path = os.path.join(generation_tmp_path, db_name, "schema.json")
    sql_schema_path = os.path.join(generation_tmp_path, db_name, "postgres.sql")

    # output paths for workload
    assert config["redset_dataset"] is not None, (
        "redset_dataset must be specified in the config"
    )
    exp_output_path = os.path.join(
        output_path,
        "generated_workloads",
        db_name,
        config["redset_dataset"],
        "cluster_" + str(instance_id),
        "database_" + str(database_id),
        experiment_name,
    )

    output_workload_path = os.path.join(exp_output_path, "workload.csv")
    output_config_path = os.path.join(exp_output_path, "used_config.json")
    plots_path = os.path.join(exp_output_path, "plots")

    if config["force_setup_creation"] or not os.path.exists(statistics_path):
        # 1. create normalized data
        create_normalized_dataset(
            csv_directory_path,
            normalized_tables_path,
            json_schema_path,
            force=config["force_setup_creation"],
        )
        # 2. create duckdb database
        create_duckdb(
            normalized_tables_path,
            output_original_db_path,
            sql_schema_path,
            force=config["force_setup_creation"],
        )
        # 3. create statistics
        create_quantiles(
            output_original_db_path,
            statistics_path,
            force=config["force_setup_creation"],
        )
    else:
        log(
            f"Statistics already exist: {statistics_path}. Skipping entire Setup Phase."
        )

    # 4. create partitioned tables
    prepare_and_scale_dataset(
        normalized_tables_path,
        split_csv_directory_path,
        output_schema_path,
        sql_schema_path,
        json_schema_path,
        statistics_path,
        schema_augmentation_factor=augmentation_factor,
    )

    # 5. create duckdb database (with new tables)
    create_duckdb(
        split_csv_directory_path, output_augmented_db_path, output_schema_path
    )

    # 6. build workload
    create_workload(
        config,
        redset_path,
        statistics_path,
        output_workload_path,
        json_schema_path,
        sql_schema_path,
        force=force_workload_creation,
        db_augmented_path=output_augmented_db_path,
    )
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=4)

    # 7. plot some results
    os.makedirs(plots_path, exist_ok=True)
    log(f"Generating plots in {plots_path}")
    create_plots(
        exp_output_path, statistics_path, redset_path, gen_strategy="generation"
    )
