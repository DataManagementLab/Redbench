import ast
import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict

import duckdb
from matching.utils import draw_bar_plot, remove_file
from utils.log import log


class Benchmark(ABC):
    def __init__(self, config, benchmark_config):
        self.benchmark_config = benchmark_config
        self.stats_db_filepath = config.stats_db_filepath
        self.benchmark_stats = None

        self.output_dir = os.path.join(config.output_dir, self.benchmark_config.id)
        os.makedirs(self.output_dir, exist_ok=True)

        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        # data shared with tmp_generation
        self.db_filepath = os.path.join(
            config.output_dir,
            "..",
            "tmp_generation",
            self.benchmark_config.id,
            "db_original.duckdb",
        )
        self.config = config

    @abstractmethod
    def _setup(self):
        assert False, "Not implemented"

    @abstractmethod
    def get_stats(self):
        assert False, "Not implemented"

    @abstractmethod
    def _compute_stats(self):
        assert False, "Not implemented"

    @abstractmethod
    def dump_plots(self):
        assert False, "Not implemented"

    @abstractmethod
    def normalize_num_joins(self, num_joins):
        assert False, "Not implemented"

    @abstractmethod
    def _setup_db(self):
        pass

    def setup(self):
        read_only = os.path.exists(self.stats_db_filepath)
        self.stats_db = duckdb.connect(self.stats_db_filepath, read_only=read_only)
        if not self.benchmark_config.override and self._is_benchmarks_setup():
            log(f"{self.benchmark_config.name} benchmark already set up.")
            return
        log(f"Setting up {self.benchmark_config.name} benchmark...")
        self._setup()


    def get_table_id(self, table_name):
        if not hasattr(self, "table_ids"):
            self.table_ids = dict()
        if table_name not in self.table_ids:
            self.table_ids[table_name.lower()] = (
                len(self.table_ids) + 1
            )  # start IDs at 1

        return self.table_ids[table_name.lower()]

    def get_table_ids(self):
        if not hasattr(self, "table_ids"):
            self._populate_benchmark_table_ids()
        return self.table_ids

    def get_table_names(self) -> Dict[int, str]:
        if not hasattr(self, "table_ids"):
            self._populate_benchmark_table_ids()
        return {v: k for k, v in self.table_ids.items()}

    def setup_db(self):
        if not self.benchmark_config.override and os.path.exists(self.db_filepath):
            log(f"{self.benchmark_config.name} database already set up.")
            return
        remove_file(self.db_filepath)
        log(
            f"Downloading and setting up the {self.benchmark_config.name} database. This may take a few minutes."
        )
        self._setup_db()

    def _cache_join_counts(self):
        self.min_num_joins = min(
            [stats["num_joins"] for stats in self.get_stats().values()]
        )
        self.max_num_joins = max(
            [stats["num_joins"] for stats in self.get_stats().values()]
        )
        self.join_counts = {stats["num_joins"] for _, stats in self.get_stats().items()}

    def compute_stats(self):
        if not self.benchmark_config.override and self._is_stats_setup():
            log(f"{self.benchmark_config.name} benchmark stats already set up.")
        else:
            log(f"Computing {self.benchmark_config.name} benchmark stats..")
            self._compute_stats()
            self._insert_table_ids()
        self._cache_join_counts()
        self._load_table_ids()

    def _insert_table_ids(self):
        stats_db_write = duckdb.connect(self.config.stats_db_filepath, read_only=False)

        stats_db_write.execute(
            f"""
            CREATE OR REPLACE TABLE {self.benchmark_config.table_ids_table} (
                table_name VARCHAR,
                table_id INTEGER
            )
        """
        )
        assert len(self.get_table_ids()) > 0, "No table IDs found."
        for table_name, table_id in self.get_table_ids().items():
            stats_db_write.execute(
                f"""
                INSERT INTO {self.benchmark_config.table_ids_table}
                VALUES ('{table_name}', {table_id})
            """
            )

    def _load_table_ids(self):
        if not hasattr(self, "table_ids"):
            self.table_ids = dict()
        rows = self.stats_db.execute(
            f"""
            SELECT * FROM {self.benchmark_config.table_ids_table}
        """
        ).fetchall()
        for table_name, table_id in rows:
            self.table_ids[table_name] = table_id

    def get_stats(self):
        if self.benchmark_stats is not None:
            return self.benchmark_stats
        self.benchmark_stats = {}
        for row in (
            # We order by filepath to ensure that Redbench generation is
            # deterministic even across different DuckDB versions.
            self.stats_db.execute(
                f"""
                SELECT * FROM {self.benchmark_config.stats_table} ORDER BY filepath
            """
            )
            .fetchdf()
            .to_dict(orient="records")
        ):
            self.benchmark_stats[row["filepath"]] = {
                "num_joins": row["num_joins"],
                "template": row["template"],
                "scanset": ast.literal_eval(row["scanset"]),
            }
        return self.benchmark_stats

    def normalize_num_joins(self, num_joins):
        normalized = (
            num_joins * (self.max_num_joins - self.min_num_joins) + self.min_num_joins
        )
        return min(self.join_counts, key=lambda v: abs(v - normalized))

    def dump_plots(self):
        dir_path = f"{self.figures_dir}/{self.benchmark_config.id}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self._dump_plots(dir_path)
        log(
            f"{self.benchmark_config.id} plots dumped to {dir_path}.",
        )

    def get_name(self):
        return self.benchmark_config.id

    def _override_stats_table(self):
        stats_db_write = duckdb.connect(self.config.stats_db_filepath, read_only=False)

        stats_db_write.execute(
            f"""
            CREATE OR REPLACE TABLE {self.benchmark_config.stats_table} (
                filepath VARCHAR,
                num_joins INTEGER,
                template VARCHAR,
                scanset VARCHAR
            )
        """
        )

    def _populate_benchmark_table_ids(self):
        raise Exception(
            "OUTDATED - we are now doing a benchmark-table-id mapping on demand. This avoids assigning IDs to tables which exist in the schema but are never queried."
        )

        # conn = duckdb.connect(self.db_filepath, read_only=True)
        # table_names = conn.execute("show tables").fetchall()
        # self.table_ids = {
        #     table_name.lower(): idx + 1
        #     for idx, (table_name,) in enumerate(sorted(table_names, key=lambda x: x[0]))
        # }
        # assert len(self.table_ids) > 0, (
        #     f"No tables found in the benchmark database {self.db_filepath}."
        # )

    def _extract_scanset_from_profile(self, node):
        scanset = []
        if node.get("operator_type") == "TABLE_SCAN":
            table = node.get("extra_info", {}).get("Table").lower()
            if table:
                scanset.append(self.get_table_id(table))
        for child in node.get("children", []):
            scanset += self._extract_scanset_from_profile(child)
        return tuple(sorted(list(set(scanset))))

    def get_db(self):
        return duckdb.connect(self.db_filepath, read_only=True)

    def _process_dir(self, dir_path, query_stats):
        template_to_num_joins = dict()
        template_to_scanset = dict()
        failed = set()
        assert os.path.exists(self.db_filepath), (
            f"Database file {self.db_filepath} does not exist."
        )
        assert len(os.listdir(dir_path)) > 0, f"No queries found in {dir_path}."
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            template = self._extract_template_from_filepath(filepath)
            if template in template_to_num_joins:
                query_stats[filepath] = {
                    "num_joins": template_to_num_joins[template],
                    "template": template,
                    "scanset": template_to_scanset[template],
                }
                continue
            if template in failed:
                continue

            # Read query
            with open(filepath, "r") as file:
                query = file.read()

            # Get number of joins in the execution plan using DuckDB Python API
            try:
                conn = duckdb.connect(self.db_filepath, read_only=True)
                result = conn.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                profile = result.fetchall()
                assert len(profile) == 1, (
                    f"Expected one result from EXPLAIN, got {len(profile)}\n{profile}"
                )
                assert profile[0][0] in ["physical_plan", "analyzed_plan"], (
                    f"Expected 'physical_plan' as the first key, got {profile[0][0]}\n{profile}"
                )
                profile = profile[0][1]
                assert isinstance(profile, str), (
                    f"Expected profile to be a string, got {type(profile)}\n{profile}"
                )

                conn.close()

                num_joins = profile.count(
                    '"operator_type": "HASH_JOIN"'
                ) - profile.count('"operator_type": "COLUMN_DATA_SCAN"')
                scanset = self._extract_scanset_from_profile(json.loads(profile))

            except Exception:
                log(f"Failed to process query \"{filepath}\". Skipping.", log_mode="error")
                failed.add(template)
                continue

            template_to_num_joins[template] = num_joins
            template_to_scanset[template] = scanset

            query_stats[filepath] = {
                "num_joins": num_joins,
                "template": template,
                "scanset": scanset,
            }

    def _insert_stats(self, filepath, query_stats):
        self.stats_db.execute(
            f"""
            INSERT INTO {self.benchmark_config.stats_table}
            VALUES (
                '{filepath}',
                {query_stats["num_joins"]},
                '{query_stats["template"]}',
                '{query_stats["scanset"]}'
            )
        """
        )

    def _bound_num_joins(self, query_stats):
        return {
            filename: query_stats[filename]
            for filename in query_stats
            if query_stats[filename]["num_joins"] >= self.min_num_joins
            and query_stats[filename]["num_joins"] <= self.max_num_joins
        }

    def _is_stats_setup(self):
        return (
            self.stats_db.execute(
                f"""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_name = '{self.benchmark_config.stats_table}'
        """
            ).fetchone()[0]
            > 0
            and self.stats_db.execute(
                f"""
            SELECT COUNT(*) FROM {self.benchmark_config.stats_table}
        """
            ).fetchone()[0]
            > 0
        )

    def _dump_plots(self, dir_path):
        stats = self.get_stats()
        map_n_joins_to_templates = defaultdict(set)
        map_n_joins_to_number_of_queries = defaultdict(int)
        for filepath, single_query_stats in stats.items():
            template = self._extract_template_from_filepath(filepath)
            map_n_joins_to_templates[single_query_stats["num_joins"]].add(template)
            map_n_joins_to_number_of_queries[single_query_stats["num_joins"]] += 1

        self._draw_bar_plot(
            map_n_joins_to_number_of_queries,
            "Number of joins",
            "Number of query instances (log scale)",
            f"Number of distinct query instances in {self.benchmark_config.name} per number of joins",
            os.path.join(dir_path, "query_instances.png"),
            log_scale_y=True,
        )
        self._draw_bar_plot(
            {k: len(map_n_joins_to_templates[k]) for k in map_n_joins_to_templates},
            "Number of joins",
            "Number of templates",
            f"Number of templates in {self.benchmark_config.name} per number of joins",
            os.path.join(dir_path, "templates.png"),
        )

        log(
            f"Number of distinct query instances in {self.benchmark_config.name}: {sum(map_n_joins_to_number_of_queries.values())}"
        )
        log(
            f"Number of templates in {self.benchmark_config.name}: {len(set(sum(list(map(list, map_n_joins_to_templates.values())), [])))}"
        )

    def _draw_bar_plot(
        self, map, x_label, y_label, title, save_path, log_scale_y=False
    ):
        tt = sorted([(p[0], p[1]) for p in map.items()], key=lambda x: x[0])
        xs = [t[0] for t in tt]
        ys = [t[1] for t in tt]
        draw_bar_plot(
            xs,
            ys,
            x_label,
            y_label,
            save_path=save_path,
            log_scale_y=log_scale_y,
            title=title,
        )
