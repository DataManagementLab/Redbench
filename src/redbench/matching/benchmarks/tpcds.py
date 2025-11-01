import os
import re

import duckdb
from matching.benchmarks.benchmark import Benchmark
from matching.utils import remove_file

QUERIES_DIR_PATH = "queries"


class TPCDSBenchmark(Benchmark):
    def __init__(self, config, benchmark_config):
        super().__init__(config, benchmark_config)
        self.queries_dir_path = os.path.join(self.output_dir, QUERIES_DIR_PATH)

    def _setup_db(self):
        conn = duckdb.connect(self.db_filepath)
        conn.execute(f"CALL dsdgen(sf = {self.benchmark_config.sf})")
        conn.close()

    def _extract_template_from_filepath(self, filepath):
        assert self.queries_dir_path in filepath, (
            f"Expected {self.queries_dir_path} in {filepath}"
        )
        return filepath.split(self.queries_dir_path)[1].split("/")[-2]

    def _replace_days_with_interval(
        self, sql
    ):  # TODO: What else? There are still a few failing templates...
        pattern = r"(\d+)\s+days"
        replacement = r"INTERVAL '\1' DAY"
        return re.sub(pattern, replacement, sql, flags=re.IGNORECASE)

    def _is_benchmarks_setup(self):
        return os.path.exists(self.queries_dir_path)

    def _setup(self, target_os="LINUX"):
        remove_file(self.queries_dir_path)

        os.system(
            f"cd {self.output_dir} && git clone https://github.com/skander-krid/tpcds-kit.git"
        )

        os.system(f"cd {self.output_dir}/tpcds-kit/tools && make OS={target_os}")
        os.system(
            f"cd {self.output_dir}/tpcds-kit/tools && ./dsdgen -scale {self.benchmark_config.sf} -verbose"
        )
        for template_num in range(1, 100):
            os.system(
                f"""
                cd {self.output_dir}/tpcds-kit/tools &&
                ./dsqgen \
                    -DIRECTORY ../query_templates \
                    -INPUT ../query_templates/templates.lst \
                    -VERBOSE Y \
                    -QUALIFY Y \
                    -SCALE {self.benchmark_config.sf} \
                    -DIALECT netezza \
                    -OUTPUT_DIR /tmp \
                    -template query{template_num}.tpl \
                    -count 1000
            """
            )

            with open("/tmp/query_0.sql", "r") as file:
                queries = file.readlines()
            os.remove("/tmp/query_0.sql")
            queries = [line.strip() for line in queries]
            queries = " ".join(queries)
            queries = self._replace_days_with_interval(queries)
            queries = queries.split(";")
            queries = [line.strip() for line in queries if len(line.strip()) > 0]
            queries = set(queries)  # Eliminate duplicates

            os.makedirs(
                os.path.join(self.queries_dir_path, str(template_num)), exist_ok=True
            )
            for i, query in enumerate(queries):
                with open(
                    f"{self.queries_dir_path}/{template_num}/{i}.sql", "w"
                ) as file:
                    file.write(query)

    def _compute_stats(self):
        self._override_stats_table()

        benchmark_stats = dict()
        for template_num in range(1, 100):
            self._process_dir(
                os.path.join(self.queries_dir_path, str(template_num)),
                benchmark_stats,
            )

        for filepath, stats in benchmark_stats.items():
            self._insert_stats(filepath, stats)
