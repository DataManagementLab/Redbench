import copy
import random
from collections import defaultdict
from typing import List

from matching.utils import (
    map_join_count_to_queries,
    map_join_count_to_templates,
    map_template_to_queries,
)
from tqdm import tqdm
from utils.load_and_preprocess_redset import get_scanset_from_redset_query
from utils.log import log


class UnsupportedDMLTypeError(Exception):
    pass


class UnknownTableMappingError(Exception):
    def __init__(self, msg=None):
        if msg is None:
            msg = "The Redset write table could not be mapped to any benchmark table."
        super().__init__(msg)


class JoinMatchingMethod:
    def __init__(self, benchmark):
        self.name = "join_matching_method"
        self.benchmark = benchmark

        benchmark_stats = benchmark.get_stats()
        self.table_names = self.benchmark.get_table_names()
        self.benchmark_stats = benchmark_stats
        self.join_count_to_queries = map_join_count_to_queries(
            benchmark_stats
        )  # TODO: Why do we need this?
        self.join_count_to_templates = map_join_count_to_templates(benchmark_stats)
        self.template_to_queries = map_template_to_queries(benchmark_stats)
        self.template_to_unused_queries = copy.deepcopy(self.template_to_queries)
        self.join_count_to_unmapped_templates = copy.deepcopy(
            self.join_count_to_templates
        )
        self.hash_to_instance = dict()
        self.scanset_to_template = dict()

    def generate_workload(self, query_timeline):
        select_timeline = [
            query for query in query_timeline if query["query_type"] == "select"
        ]

        min_redset_join_count = (
            min([query["num_joins"] for query in select_timeline])
            if len(select_timeline) > 0
            else 0
        )
        max_redset_join_count = (
            max([query["num_joins"] for query in select_timeline])
            if len(select_timeline) > 0
            else 0
        )
        assert (
            min_redset_join_count != max_redset_join_count or max_redset_join_count == 0
        ), f"Redset join counts are all the same ({min_redset_join_count})."

        # Match the SELECT queries first
        workload = []
        sampling_steps_occs = defaultdict(int)
        for redset_query in tqdm(query_timeline, desc="Matching SELECT queries first"):
            if redset_query["query_type"] != "select":
                continue
            workload_query, sampling_step = self._match_select_query(
                redset_query, min_redset_join_count, max_redset_join_count
            )
            workload_query["redset_query"] = redset_query
            workload.append(workload_query)
            sampling_steps_occs[sampling_step] += 1

        # Now match the DMLs
        self._add_dmls_to_workload(workload, query_timeline)

        # Compute stats
        stats = {"sampling_steps": sampling_steps_occs}

        return workload, stats

    def _get_tables_order(self, scansets):
        table_occs = defaultdict(int)
        for scanset in scansets:
            assert len(set(scanset)) == len(scanset), (
                f"Scanset {scanset} contains duplicates."
            )
            for table in scanset:
                table_occs[table] += 1
        table_occs = sorted(table_occs.items(), key=lambda x: x[1], reverse=True)
        return list(map(lambda x: x[0], table_occs))

    def _add_dmls_to_workload(self, workload: List, query_timeline):
        # Order tables by their occurrences in the Redset and matched workload respectively
        redset_table_occs = self._get_tables_order(
            [
                get_scanset_from_redset_query(query)
                for query in query_timeline
                if query["query_type"] == "select"
            ]
        )
        benchmark_table_occs = self._get_tables_order(
            [stats["scanset"] for stats in self.benchmark.get_stats().values()]
        )

        # Match the tables
        table_mapping = dict()
        for idx in range(min(len(redset_table_occs), len(benchmark_table_occs))):
            table_mapping[redset_table_occs[idx]] = benchmark_table_occs[idx]

        table_names = self.benchmark.get_table_names()
        idx = 0
        dml_mapping_error_ctr = defaultdict(int)
        for query in tqdm(query_timeline, desc="Matching DML queries"):
            if query["query_type"] == "select":
                idx += 1
                continue
            try:
                matched_dml = self._match_dml_query(
                    query,
                    table_mapping,
                    table_names,
                    pick_random_table_if_unmapped=True,
                )
            except UnsupportedDMLTypeError:
                dml_mapping_error_ctr["unsupported dml type"] += 1
                continue
            except UnknownTableMappingError:
                dml_mapping_error_ctr["unknown table mapping"] += 1
                continue

            matched_dml["redset_query"] = query
            workload.insert(idx, matched_dml)
            idx += 1

        log(f"DML mapping errors: {dml_mapping_error_ctr}")

    def _match_dml_query(
        self,
        redset_query,
        table_mapping,
        table_names,
        pick_random_table_if_unmapped=False,
    ):
        if redset_query["query_type"] not in ["insert", "update", "delete"]:
            raise UnsupportedDMLTypeError()
        write_table = int(redset_query["write_table_ids"])
        if write_table not in table_mapping:
            if pick_random_table_if_unmapped:
                key, benchmark_write_table = random.choice(list(table_names.items()))

                # add to mapping for future use
                table_mapping[write_table] = key
            else:
                raise UnknownTableMappingError(write_table)
        else:
            benchmark_write_table = table_names[table_mapping[write_table]]
        query = {
            "query_type": redset_query["query_type"],
            "write_table_id": write_table,
            "benchmark_write_table": benchmark_write_table,
            "arrival_timestamp": redset_query["arrival_timestamp"].isoformat(),
        }

        return query

    def _match_select_query(
        self, redset_query, min_redset_join_count, max_redset_join_count
    ):
        assert redset_query["query_type"] == "select", (
            f"Expected a SELECT query, but got {redset_query['query_type']}."
        )

        # Normalize join count
        join_diff = max_redset_join_count - min_redset_join_count
        if join_diff == 0:
            scaled_num_joins = self.benchmark.normalize_num_joins(0.0)
        else:
            scaled_num_joins = self.benchmark.normalize_num_joins(
                (redset_query["num_joins"] - min_redset_join_count)
                / (max_redset_join_count - min_redset_join_count)
            )
        redset_query = copy.deepcopy(redset_query)

        redset_query_hash, num_joins = redset_query["query_hash"], scaled_num_joins
        redset_query_scanset = get_scanset_from_redset_query(redset_query)
        benchmark_query = None

        if redset_query_hash in self.hash_to_instance:
            # We have seen the same query hash before
            benchmark_query = copy.deepcopy(self.hash_to_instance[redset_query_hash])
            benchmark_query["arrival_timestamp"] = redset_query[
                "arrival_timestamp"
            ].isoformat()
            return benchmark_query, "1"

        def step_6():
            benchmark_query = None
            templates_pool = copy.deepcopy(self.join_count_to_templates[num_joins])
            random.shuffle(templates_pool)
            for template in templates_pool:
                # (6): Pick a random, already mapped, CEB+ template with unused query instances
                if (
                    template not in self.join_count_to_unmapped_templates
                    and len(self.template_to_unused_queries[template]) > 0
                ):
                    benchmark_query = self.template_to_unused_queries[template].pop()
                    final_step = "6"
                    break
            # (7): No already mapped CEB+ templates with remaining query instances
            #  -> just pick a random query instance
            if benchmark_query is None:
                final_step = "6"  # For the plots, we will count this as step 6
                benchmark_query = random.choice(self.join_count_to_queries[num_joins])
            return benchmark_query, final_step

        # We have already encountered this readset (1)
        if redset_query_scanset in self.scanset_to_template:
            corresponding_ceb_template = self.scanset_to_template[redset_query_scanset]
            remaining_ceb_query_instances_for_template = (
                self.template_to_unused_queries[corresponding_ceb_template]
            )
            if len(remaining_ceb_query_instances_for_template) > 0:
                used_sampling_step = "2"
                benchmark_query = (
                    remaining_ceb_query_instances_for_template.pop()
                )  # (2)
            else:
                benchmark_query, final_step = step_6()
                used_sampling_step = f"3 -> {final_step}"  # (3)
        # This readset has never occured before (4)
        else:
            if len(self.join_count_to_unmapped_templates[num_joins]) > 0:
                # We still have unmapped CEB+ templates
                # Look for the one with the most number of remaining queries
                best, best_value = None, 0
                for candidate_template in self.join_count_to_unmapped_templates[
                    num_joins
                ]:
                    this_value = len(
                        self.template_to_unused_queries[candidate_template]
                    )
                    if this_value > best_value:
                        best_value = this_value
                        best = candidate_template
                if best_value > 0:
                    # We found one with some remaining queries
                    corresponding_ceb_template = best
                    assert (
                        len(self.template_to_unused_queries[corresponding_ceb_template])
                        > 0
                    )

                    # Mark the CEB+ template as mapped/ remove it from the unmapped list
                    join_counts = set()
                    for (
                        join_count,
                        unmapped_templates_list,
                    ) in self.join_count_to_unmapped_templates.items():
                        if corresponding_ceb_template in unmapped_templates_list:
                            unmapped_templates_list.remove(corresponding_ceb_template)
                            join_counts.add(join_count)
                    assert len(join_counts) == 1, (
                        f"The same template {corresponding_ceb_template} of {self.benchmark.benchmark_config.name} produces different num_joins: {join_counts}"
                    )

                    # Add the mapping readset -> CEB+ template
                    self.scanset_to_template[redset_query_scanset] = (
                        corresponding_ceb_template
                    )

                    # Use one of the unused query instances
                    benchmark_query = self.template_to_unused_queries[
                        corresponding_ceb_template
                    ].pop()
                    used_sampling_step = "5"
            # All CEB+ templates have already been mapped (6)
            if benchmark_query is None:
                benchmark_query, final_step = step_6()
                used_sampling_step = f"4 -> {final_step}"

        benchmark_query = {
            "filepath": benchmark_query,
            "scanset": redset_query_scanset,
            "benchmark_scanset": [
                self.table_names[table]
                for table in self.benchmark_stats[benchmark_query]["scanset"]
            ],
            "join_count": redset_query["num_joins"],
            "benchmark_join_count": self.benchmark_stats[benchmark_query]["num_joins"],
            "arrival_timestamp": redset_query["arrival_timestamp"].isoformat(),
            "query_type": redset_query["query_type"],
            "query_hash": redset_query_hash,
        }
        self.hash_to_instance[redset_query_hash] = benchmark_query
        return benchmark_query, used_sampling_step
