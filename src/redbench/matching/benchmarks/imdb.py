import os
import re

from download_artifacts import download_artifacts
from matching.benchmarks.benchmark import Benchmark
from matching.utils import get_sub_directories
from utils.log import log

JOB_QUERIES_DIR_PATH = "benchmarks/job"
CEB_QUERIES_DIR_PATH = "benchmarks/ceb"
RAW_DATA_DIR_PATH = "raw_data"


class IMDbBenchmark(Benchmark):
    """
    This class represents the IMDb benchmarks JOB and CEB.
    Query stats are computed for both benchmarks.
    """

    def __init__(self, config, benchmark_config):
        super().__init__(config, benchmark_config)
        self.job_queries_dir_path = os.path.join(self.output_dir, JOB_QUERIES_DIR_PATH)
        self.ceb_queries_dir_path = os.path.join(self.output_dir, CEB_QUERIES_DIR_PATH)
        self.raw_data_path = os.path.join(self.output_dir, RAW_DATA_DIR_PATH)

    def _is_benchmarks_setup(self):
        return os.path.exists(self.job_queries_dir_path) and os.path.exists(
            self.ceb_queries_dir_path
        )

    def _setup(self):
        os.system(f"rm -rf {self.output_dir}/benchmarks")
        os.system(
            f"tar -xzf src/redbench/matching/benchmarks/imdb/queries.tar.gz -C {self.output_dir}/"
        )

    def _setup_db(self):
        artifacts_dir = os.path.join(self.config.output_dir, "..")
        if not os.path.exists(
            os.path.join(artifacts_dir, "tmp_generation", "imdb", "db_original.duckdb")
        ):
            # log("Downloading IMDb artifacts from OSF...")
            download_artifacts(
                artifacts_dir=artifacts_dir,
                databases=["imdb"],
                download_only_duckdb_file=True,
            )

    # def _setup_db_from_cwi(self):
    #     os.makedirs(self.raw_data_path, exist_ok=True)

    #     os.system("wget -q http://event.cwi.nl/da/job/imdb.tgz -O /tmp/imdb.tgz")
    #     os.system(f"tar -xzf /tmp/imdb.tgz -C {self.raw_data_path}")
    #     os.system("rm /tmp/imdb.tgz")

    #     schema_filepath = "imdb/schema.sql"
    #     load_filepath = "imdb/load.sql"

    #     # Update raw file paths in load.sql : replace 'xyz.csv' with 'raw_data/xyz.csv' for any xyz
    #     with open(load_filepath, "r") as file:
    #         load_sql = file.read()
    #     load_sql = re.sub(
    #         r"'(?!path/)([^']+\.csv)'",
    #         lambda m: f"'{os.path.join(self.raw_data_path, m.group(1))}'",
    #         load_sql,
    #     )
    #     load_filepath = "/tmp/imdb_schema.sql"
    #     with open(load_filepath, "w") as file:
    #         file.write(load_sql)

    #     # create duckdb connection
    #     conn = duckdb.connect(self.db_filepath, read_only=False)

    #     # Execute schema file
    #     with open(schema_filepath, "r") as f:
    #         schema_sql = f.read()
    #         conn.execute(schema_sql)

    #     # Execute load file
    #     with open(load_filepath, "r") as f:
    #         load_sql = f.read()
    #         conn.execute(load_sql)

    #     conn.close()

    def _extract_template_from_filepath(self, filepath):
        filepath = filepath.split("imdb/benchmarks/")[
            1
        ]
        benchmark = filepath.split("/")[0]
        template = filepath.split("/")[1]
        if benchmark == "job":
            # Get the number as the template (1, 2, 3, ..., 33)
            return re.match(r"\d+", template).group()
        assert benchmark == "ceb"
        # The folder name is the template (1a, 2a, 2b, ..., 11b)
        return filepath.split("/")[1]

    def _compute_stats(self):
        self._override_stats_table()
        benchmark_stats = dict()

        log("Collecting stats for JOB queries..")
        self._process_dir(self.job_queries_dir_path, benchmark_stats)

        log("Collecting stats for CEB queries..")
        for subdir in get_sub_directories(self.ceb_queries_dir_path):
            self._process_dir(subdir, benchmark_stats)

        assert len(benchmark_stats) > 0, "No benchmark stats collected."
        for filepath, stats in benchmark_stats.items():
            self._insert_stats(filepath, stats)
