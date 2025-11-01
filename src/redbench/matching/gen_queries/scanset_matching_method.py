import copy
import random
from collections import defaultdict
from typing import Dict

from matching.gen_queries.scanset_mapper import ScansetMapper
from utils.load_and_preprocess_redset import get_scanset_from_redset_query


class ScansetMatchingMethod:
    def __init__(self, benchmark, config: Dict):
        self.name = "scanset_matching_method"
        self.benchmark = benchmark
        self.benchmark_stats = benchmark.get_stats()
        self.benchmark_scansets = [
            stats["scanset"] for stats in self.benchmark_stats.values()
        ]
        self.config = config

    def _add_dmls_to_workload(self, workload, query_timeline, scanset_mapper):
        idx = 0

        cannot_map_write_table_ctr = 0
        for redset_query in query_timeline:
            if redset_query["query_type"] == "select":
                # increment idx to insert DMLs at the right position
                idx += 1
                continue
            if redset_query["query_type"] not in ["insert", "update", "delete"]:
                continue
            benchmark_versioned_write_table = (
                scanset_mapper.translate_versioned_redset_table(
                    int(redset_query["write_table_ids"]),
                    pick_random_table_if_not_mapped=True,
                )
            )
            if benchmark_versioned_write_table is None:
                cannot_map_write_table_ctr += 1
                continue
            matched_dml = {
                "query_type": f"{redset_query['query_type']}",
                "write_table_id": int(redset_query["write_table_ids"]),
                "benchmark_write_table": benchmark_versioned_write_table,
                "arrival_timestamp": redset_query["arrival_timestamp"].isoformat(),
            }
            matched_dml["redset_query"] = redset_query
            workload.insert(idx, matched_dml)
            idx += 1

        print(
            f"Could not map {cannot_map_write_table_ctr} DML queries to benchmark tables (mapping for the write table not known)."
        )
        return cannot_map_write_table_ctr

    def _cleanup_table_versions(self, workload):
        # cleanup table version and transalte table-id to actual table name

        # lookup table names
        table_names = self.benchmark.get_table_names()

        if self.config["use_table_versioning"]:
            table_versions = defaultdict(set)
            for query in workload:
                assert ("versioning" in query) != ("benchmark_write_table" in query)
                if "versioning" in query:
                    query_tables = query["versioning"].values()
                else:
                    query_tables = [query["benchmark_write_table"]]
                for table in query_tables:
                    base_table, version = table.rsplit("_", 1)
                    table_versions[base_table].add(version)

            # Compactify the versions range for each table.
            # E.g. T1 has versions {1, 2, 5} -> {1, 2, 3}
            new_versions = dict()
            for table, versions in table_versions.items():
                versions = sorted(versions)
                for idx, version in enumerate(versions):
                    new_versions[f"{table}_{version}"] = (
                        f"{table_names[int(table)]}_{idx + 1}"
                    )

        for query in workload:
            if self.config["use_table_versioning"]:
                # translate all tables using versioning
                if "versioning" in query:
                    query["versioning"] = {
                        table_names[int(k)]: new_versions[v]
                        for k, v in query["versioning"].items()
                    }
                else:
                    query["benchmark_write_table"] = new_versions[
                        query["benchmark_write_table"]
                    ]
            else:
                # translate only benchmark write table
                if "benchmark_write_table" in query:
                    query["benchmark_write_table"] = table_names[
                        query["benchmark_write_table"]
                    ]
            if "benchmark_scanset" in query:
                query["benchmark_scanset"] = [
                    table_names[table] for table in query["benchmark_scanset"]
                ]

        if self.config["use_table_versioning"]:
            return {
                table_names[int(table)]: len(versions)
                for table, versions in table_versions.items()
            }
        else:
            return None

    def _get_scanset_mapper(self, query_timeline):
        redset_scansets = [
            get_scanset_from_redset_query(query)
            for query in query_timeline
            if query["query_type"] == "select"
        ]
        return ScansetMapper(
            config=self.config,
            redset_scansets=redset_scansets,
            benchmark_scansets=self.benchmark_scansets,
        )

    def generate_workload(self, query_timeline):
        select_timeline = [
            query for query in query_timeline if query["query_type"] == "select"
        ]
        if select_timeline:
            scanset_mapper = self._get_scanset_mapper(query_timeline)

        scanset_to_unmapped_instances = defaultdict(list)
        scanset_to_all_instances = defaultdict(list)
        for filepath, stats in self.benchmark_stats.items():
            scanset_to_unmapped_instances[stats["scanset"]].append(filepath)
            scanset_to_all_instances[stats["scanset"]].append(filepath)
        scanset_to_unmapped_instances = dict(scanset_to_unmapped_instances)
        scanset_to_all_instances = dict(scanset_to_all_instances)

        workload = []
        hash_to_instance = dict()
        not_enough_instances = 0
        were_cached = 0
        for redset_query in query_timeline:
            if (
                redset_query["query_type"] != "select"
            ):  # Here we only map SELECT queries
                continue

            # We can (and must) answer this query from the cache
            if redset_query["query_hash"] in hash_to_instance:
                benchmark_query = copy.deepcopy(
                    hash_to_instance[redset_query["query_hash"]]
                )
                benchmark_query["arrival_timestamp"] = redset_query[
                    "arrival_timestamp"
                ].isoformat()
                benchmark_query["redset_query"] = redset_query
                workload.append(benchmark_query)
                were_cached += 1
                continue

            # Find the corresponding benchmark scanset
            redset_query_scanset = get_scanset_from_redset_query(redset_query)
            benchmark_scanset, versioning, distance = (
                scanset_mapper.find_closest_benchmark_scanset(redset_query_scanset)
            )

            if scanset_to_unmapped_instances[benchmark_scanset]:
                # We have unused queries for this scanset, so we can use one of them
                benchmark_query_filepath = scanset_to_unmapped_instances[
                    benchmark_scanset
                ].pop()
            else:
                # We have no unused queries for this scanset, so we need to pick one randomly
                benchmark_query_filepath = random.choice(
                    scanset_to_all_instances[benchmark_scanset]
                )
                not_enough_instances += 1

            benchmark_query = {
                "scanset": redset_query_scanset,
                "benchmark_scanset": benchmark_scanset,
                "filepath": benchmark_query_filepath,
                "query_hash": redset_query["query_hash"],
                "distance": distance,
                "join_count": redset_query["num_joins"],
                "benchmark_join_count": self.benchmark_stats[benchmark_query_filepath][
                    "num_joins"
                ],
                "arrival_timestamp": redset_query["arrival_timestamp"].isoformat(),
                "query_type": redset_query["query_type"],
            }

            if self.config["use_table_versioning"]:
                benchmark_query["versioning"] = versioning

            hash_to_instance[redset_query["query_hash"]] = benchmark_query
            benchmark_query["redset_query"] = redset_query
            workload.append(benchmark_query)

        assert len(select_timeline) == len(workload), (
            f"{len(select_timeline)} vs {len(workload)}"
        )

        cannot_map_write_table_ctr = self._add_dmls_to_workload(
            workload, query_timeline, scanset_mapper
        )
        version_counts = self._cleanup_table_versions(workload)

        stats = {
            "not_enough_instances": not_enough_instances,
            "table_version_counts": version_counts,
            "scanset_mapper_stats": scanset_mapper.get_stats(),
            "cannot_map_write_table_ctr": cannot_map_write_table_ctr,
        }

        return workload, stats
