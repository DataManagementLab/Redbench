import csv
import json
import os
import random

from generation.run import gen_expname_from_config
from matching.gen_queries.join_matching_method import JoinMatchingMethod
from matching.gen_queries.matching_utils import SimpleDmlsConstructor, init_query
from matching.gen_queries.scanset_matching_method import ScansetMatchingMethod
from matching.utils import (
    get_query_timeline,
    get_sub_directories,
)
from plots.create_plots import create_plots
from tqdm import tqdm
from utils.load_and_preprocess_redset import get_scanset_from_redset_query
from utils.log import log


class QueryGenerator:
    def __init__(self, benchmark, config, overwrite_existing: bool):
        serializable_config = vars(config).copy()
        if "stats_db" in serializable_config:
            serializable_config.pop("stats_db")
        serializable_config["support_benchmarks"] = serializable_config[
            "support_benchmarks"
        ][0].__dict__

        self.config = config
        self.serializable_config = serializable_config
        self.benchmark = benchmark
        self.overwrite_existing = overwrite_existing

        # extract matching method
        if config.matching_method == "join":
            self.matching_method = JoinMatchingMethod(benchmark)
        elif config.matching_method == "scanset":
            self.matching_method = ScansetMatchingMethod(benchmark, vars(config))
        else:
            raise ValueError(
                f"Unknown matching method {config.matching_method}. Expected: ['join', 'scanset']."
            )

        # assemble workload dir

        workload_name = gen_expname_from_config(
            serializable_config, strategy="matching"
        )
        self.workloads_dir = os.path.join(
            self.config.wl_dir,
            benchmark.get_name(),
            config.redset_dataset,
            "cluster_" + str(self.config.cluster_id),
            "database_" + str(self.config.database_id),
            workload_name,
        )

    def exists(self):
        return (
            os.path.exists(self.workloads_dir)
            and os.path.exists(os.path.join(self.workloads_dir, "workload.csv"))
            and len(os.listdir(self.workloads_dir)) > 0
            and all(
                map(
                    lambda subdir: any(
                        map(lambda file: file.endswith(".sql"), os.listdir(subdir))
                    ),
                    get_sub_directories(self.workloads_dir),
                )
            )
        )

    def _query_matched_query_hash(self, query):
        filepath, versioning = tuple(query["filepath"]), query.get("versioning")
        return (
            tuple([filepath])
            if not versioning
            else (filepath, tuple(sorted((k, v) for k, v in versioning.items())))
        )

    def _get_versioned_benchmark_scanset(self, query):
        if "versioning" in query:
            assert query["versioning"] is not None
            return tuple(
                query["versioning"].get(table, table)
                for table in query["benchmark_scanset"]
            )
        return tuple(query["benchmark_scanset"])

    def _compute_repetition_ratio(self, queries, is_dml_aware, is_original, is_scanset):
        last_select = dict()
        last_write = dict()
        rep = tot = 0
        for i, q in enumerate(queries):
            t = q["query_type"]
            if t == "select":
                tot += 1
                scanset = (
                    get_scanset_from_redset_query(q)
                    if is_original
                    else self._get_versioned_benchmark_scanset(q)
                )
                if is_scanset:
                    h = scanset
                else:
                    h = (
                        q["query_hash"]
                        if is_original
                        else self._query_matched_query_hash(q)
                    )
                if h in last_select:
                    ok = True
                    for table in scanset:
                        if last_write.get(table, -1) > last_select[h]:
                            ok = False
                            break
                    if ok:
                        rep += 1
                last_select[h] = i
            else:
                table = (
                    int(q["write_table_ids"])
                    if is_original
                    else q["benchmark_write_table"]
                )
                if table is not None and is_dml_aware:
                    last_write[table] = i
        return rep / tot if tot else 0.0

    def generate(self):
        random.seed(0)  # For reproducibility

        # Override the existing workload if needed
        if not self.overwrite_existing and self.exists():
            log(
                f"Redbench already generated for {self.benchmark.benchmark_config.id}.",
                log_mode="warning",
            )
            return

        os.system(f"rm -rf {self.workloads_dir}")
        log(
            f"Generating redbench[{self.benchmark.get_name()}] with {self.matching_method.name}"
            + f" for cluster {self.config.cluster_id}, database {self.config.database_id}.."
        )

        # Get the Redset query timeline
        redset_exclude_tables_never_read = getattr(
            self.config, "redset_exclude_tables_never_read", False
        )
        query_timeline = get_query_timeline(
            self.config.redset_path,
            self.config.cluster_id,
            self.config.database_id,
            self.config.start_date,
            self.config.end_date,
            redset_exclude_tables_never_read=redset_exclude_tables_never_read,
            limit_redset_rows_read=self.config.limit_redset_rows_read,
        )

        print(f"Loaded {len(query_timeline)} queries from Redset.")

        # Generate and write the workload
        workload, stats = self.matching_method.generate_workload(query_timeline)

        # Add workload stats
        stats.update(
            {
                "matched_workload_size": {
                    "num_queries": len(workload),
                    "num_select_queries": len(
                        [q for q in workload if q["query_type"] == "select"]
                    ),
                    "num_insert_queries": len(
                        [q for q in workload if q["query_type"] == "insert"]
                    ),
                    "num_update_queries": len(
                        [q for q in workload if q["query_type"] == "update"]
                    ),
                    "num_delete_queries": len(
                        [q for q in workload if q["query_type"] == "delete"]
                    ),
                    "repetition_ratio": self._compute_repetition_ratio(
                        workload,
                        is_dml_aware=False,
                        is_original=False,
                        is_scanset=False,
                    ),
                    "dml_aware_repetition_ratio": self._compute_repetition_ratio(
                        workload, is_dml_aware=True, is_original=False, is_scanset=False
                    ),
                    "scanset_repetition_ratio": self._compute_repetition_ratio(
                        workload, is_dml_aware=False, is_original=False, is_scanset=True
                    ),
                    "dml_aware_scanset_repetition_ratio": self._compute_repetition_ratio(
                        workload, is_dml_aware=True, is_original=False, is_scanset=True
                    ),
                },
                "timeline_size": {
                    "num_queries": len(query_timeline),
                    "num_select_queries": len(
                        [q for q in query_timeline if q["query_type"] == "select"]
                    ),
                    "num_insert_queries": len(
                        [q for q in query_timeline if q["query_type"] == "insert"]
                    ),
                    "num_update_queries": len(
                        [q for q in query_timeline if q["query_type"] == "update"]
                    ),
                    "num_delete_queries": len(
                        [q for q in query_timeline if q["query_type"] == "delete"]
                    ),
                    "repetition_ratio": self._compute_repetition_ratio(
                        query_timeline,
                        is_dml_aware=False,
                        is_original=True,
                        is_scanset=False,
                    ),
                    "dml_aware_repetition_ratio": self._compute_repetition_ratio(
                        query_timeline,
                        is_dml_aware=True,
                        is_original=True,
                        is_scanset=False,
                    ),
                    "scanset_repetition_ratio": self._compute_repetition_ratio(
                        query_timeline,
                        is_dml_aware=False,
                        is_original=True,
                        is_scanset=True,
                    ),
                    "dml_aware_scanset_repetition_ratio": self._compute_repetition_ratio(
                        query_timeline,
                        is_dml_aware=True,
                        is_original=True,
                        is_scanset=True,
                    ),
                },
                "cluster_id": self.config.cluster_id,
                "database_id": self.config.database_id,
            }
        )
        for query in workload:
            query["redset_query"]["arrival_timestamp"] = query["redset_query"][
                "arrival_timestamp"
            ].isoformat()

        self._write_result_files_to_disk(workload, stats)

    def create_plots(self):
        exp_output_path = self.workloads_dir
        log(f"Generating plots in {exp_output_path}")
        create_plots(
            exp_output_path, None, self.config.redset_path, gen_strategy="matching"
        )

    def _write_workload_file_to_disk(self, sampled_benchmark):
        output_path = os.path.join(self.workloads_dir, "queries.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(sampled_benchmark, file, indent=2)
        log(f"Redbench workload written to {output_path}")

    def _write_config_file_to_disk(self, config):
        output_path = os.path.join(self.workloads_dir, "used_config.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(config, file, indent=2)
        log(f"Redbench config written to {output_path}")

    def _write_stats_file_to_disk(self, stats):
        output_path = os.path.join(self.workloads_dir, "stats.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(stats, file, indent=2)
        log(f"Redbench stats written to {output_path}")

    def _write_result_files_to_disk(self, workload, stats):
        # dump the workload and stats to disk as json
        self._write_workload_file_to_disk(workload)
        self._write_stats_file_to_disk(stats)
        self._write_config_file_to_disk(self.serializable_config)

        # convert the format to csv format
        csv_data = [OUTPUT_CSV_HEADER]
        nCols = len(OUTPUT_CSV_HEADER)
        workload.sort(key=lambda q: q["arrival_timestamp"])
        dml_constructor = SimpleDmlsConstructor(self.benchmark.get_db())
        for query in tqdm(workload, desc="Assembling select & dml queries"):
            csv_row = [None] * nCols
            redset_query = query.get("redset_query", None)
            assert redset_query is not None

            for col in REDSET_HEADER:
                val = redset_query.get(col, None)
                csv_row[get_col_index(col)] = val

            # this is just a placeholder
            csv_row[get_col_index("structural_repetition_id")] = 1
            csv_row[get_col_index("unique_db_instance")] = 0

            query_type = query["query_type"]
            if query_type == "select":
                query_filepath = query["filepath"]
                table_versions = query.get("versioning", None)
                csv_row[get_col_index("sql")] = init_query(
                    query_filepath, table_versions, remove_linebreaks=True
                )
                csv_row[get_col_index("join_tables")] = (
                    ",".join(query["versioning"].values())
                    if table_versions is not None
                    else ",".join(query["benchmark_scanset"])
                )
                csv_row[get_col_index("read_tables")] = ",".join(
                    self._get_versioned_benchmark_scanset(query)
                )
                csv_row[get_col_index("exact_repetition_hash")] = query["query_hash"]
            else:
                csv_row[get_col_index("write_table")] = query["benchmark_write_table"]
                csv_row[get_col_index("sql")] = dml_constructor(query)
                csv_row[get_col_index("exact_repetition_hash")] = query["redset_query"][
                    "query_hash"
                ]

            csv_data.append(csv_row)

        csv_output_path = os.path.join(self.workloads_dir, "workload.csv")
        with open(csv_output_path, "w") as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerows(csv_data)

        # create the plots
        # FIXME rebase after PR merge
        # plot_repetition(csv_output_path, os.path.join(self.workloads_dir, "repetition.png"))
        # plot_query_runs(csv_output_path, os.path.join(self.workloads_dir, "query_runs.png"))
        # plot_redset_comparison # FIXME


REDSET_HEADER = [
    "instance_id",
    "cluster_size",
    "user_id",
    "database_id",
    "query_id",
    "arrival_timestamp",
    "compile_duration_ms",
    "queue_duration_ms",
    "execution_duration_ms",
    "feature_fingerprint",
    "was_aborted",
    "was_cached",
    "cache_source_query_id",
    "query_type",
    "num_permanent_tables_accessed",
    "num_external_tables_accessed",
    "num_system_tables_accessed",
    "read_table_ids",
    "write_table_ids",
    "mbytes_scanned",
    "mbytes_spilled",
    "num_joins",
    "num_scans",
    "num_aggregations",
]

ADDED_COLUMNS = [
    "start_t",
    "read_tables",
    "join_tables",
    "write_table",
    "structural_repetition_id",
    "unique_db_instance",
    "exact_repetition_hash",
    "sql",
    "approximated_scan_selectivities",
]

OUTPUT_CSV_HEADER = REDSET_HEADER + ADDED_COLUMNS


def get_col_index(col_name) -> int:
    try:
        ret = OUTPUT_CSV_HEADER.index(col_name)
        assert isinstance(ret, int)
        return ret
    except ValueError:
        raise ValueError(f"Column {col_name} not found in OUTPUT_CSV_HEADER.")
