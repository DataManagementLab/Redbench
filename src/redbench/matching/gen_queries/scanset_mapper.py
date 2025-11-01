import os
import random
import statistics
import time
from collections import Counter, defaultdict

import hillmapper
from utils.log import log


class ScansetMapper:
    def __init__(self, config, redset_scansets, benchmark_scansets):
        # Validate input scansets
        ScansetMapper._validate_input_scansets(redset_scansets)
        ScansetMapper._validate_input_scansets(benchmark_scansets)

        # Clean up scansets
        redset_scansets = ScansetMapper._clean_up_scansets(redset_scansets)
        benchmark_scansets = ScansetMapper._clean_up_scansets(benchmark_scansets)

        # Translate Redset scansets to remove non-existing tables.
        # E.g.: scansets = [(1, 2), (4, 5)] -> [(1, 2), (3, 4)]
        redset_scansets, self.translation = ScansetMapper._translate_scansets(
            redset_scansets
        )

        # Remove duplicates from benchmark scansets
        self.benchmark_scansets = set(benchmark_scansets)

        # Validate cleaned up scansets
        ScansetMapper._validate_clean_scansets(redset_scansets)
        ScansetMapper._validate_clean_scansets(self.benchmark_scansets)

        self.table_mapping = self.match_tables(
            redset_scansets, self.benchmark_scansets, config
        )

        self.config = config
        self.versioned_table_mapping = dict()
        benchmark_table_counts = defaultdict(int)
        for redset_table, benchmark_table in self.table_mapping.items():
            if self.config["use_table_versioning"]:
                # only increment the count if the table is versioned - if not: it will always use version 0
                benchmark_table_counts[benchmark_table] += 1

                versioned_benchmark_table = (
                    f"{benchmark_table}_{benchmark_table_counts[benchmark_table]}"
                )
                self.versioned_table_mapping[redset_table] = versioned_benchmark_table
            else:
                # mapping of number to number (no versioning, and not yet translated back to original table name)
                self.versioned_table_mapping[redset_table] = benchmark_table

    def translate_redset_table(self, redset_table):
        if redset_table not in self.translation:  # Table written to but never read from
            return None
        redset_table = ScansetMapper._apply_translation(
            [redset_table], self.translation
        )[0]
        benchmark_table = ScansetMapper._apply_translation(
            [redset_table], self.table_mapping
        )[0]
        return benchmark_table

    def translate_versioned_redset_table(
        self, redset_table, pick_random_table_if_not_mapped: bool = False
    ):
        if redset_table not in self.translation:
            if pick_random_table_if_not_mapped:
                # these are the tables that were never read from, so we can pick any versioned table
                return random.choice(list(self.versioned_table_mapping.values()))
            return None
        redset_table = ScansetMapper._apply_translation(
            [redset_table], self.translation
        )[0]
        benchmark_table = ScansetMapper._apply_translation(
            [redset_table], self.versioned_table_mapping
        )[0]
        return benchmark_table

    def find_closest_benchmark_scanset(self, redset_scanset):
        redset_scanset = ScansetMapper._apply_translation(
            redset_scanset, self.translation
        )
        redset_scanset = ScansetMapper._clean_up_scansets([redset_scanset])[0]
        mapped_scanset = ScansetMapper._apply_translation(
            redset_scanset, self.table_mapping
        )
        mapped_scanset = ScansetMapper._clean_up_scansets([mapped_scanset])[0]

        closest_distance = float("inf")
        for benchmark_scanset_candidate in set(self.benchmark_scansets):
            distance = ScansetMapper._hamming_distance(
                benchmark_scanset_candidate, mapped_scanset
            )
            if distance < closest_distance:
                closest_distance = distance
                benchmark_scanset = benchmark_scanset_candidate

        versioning = dict()
        redset_scanset = list(redset_scanset)
        random.shuffle(redset_scanset)  # Randomize the order of tables in the scanset
        for redset_table in redset_scanset:
            # If the image of a redset table appears in the benchmark scanset,
            # and the table is versioned, we need to add the versioning information.
            # If multiple redset versioned tables map to the same benchmark table,
            # we choose one randomly (because of the shuffle above).
            benchmark_table = ScansetMapper._apply_translation(
                [redset_table], self.table_mapping
            )[0]
            if benchmark_table in benchmark_scanset:
                versioning[benchmark_table] = self.versioned_table_mapping[redset_table]
        return benchmark_scanset, versioning, closest_distance

    @staticmethod
    def _hamming_distance(scanset1, scanset2):  # Hamming distance between both scansets
        s1, s2 = Counter(scanset1), Counter(scanset2)
        keys = s1.keys() | s2.keys()
        return sum(abs(s1[k] - s2[k]) for k in keys)

    @staticmethod
    def _count_tables(scansets):
        return max([max(scanset) for scanset in scansets if scanset])

    # Remove non-existing tables.
    @staticmethod
    def _translate_scansets(scansets):
        tables = sorted(list(set([table for scanset in scansets for table in scanset])))
        conversion = dict()
        for idx, table in enumerate(tables):
            conversion[table] = idx + 1
        return [
            tuple(sorted(list({conversion[table] for table in scanset})))
            for scanset in scansets
        ], conversion

    @staticmethod
    def _apply_translation(scanset, translation):
        return tuple(sorted([translation[table] for table in scanset]))

    @staticmethod
    def _clean_up_scansets(scansets):
        scansets = map(sorted, scansets)
        scansets = map(tuple, scansets)
        scansets = filter(lambda x: len(x) > 0, scansets)
        scansets = list(scansets)
        return scansets

    @staticmethod
    def _validate_clean_scansets(scansets):
        # Make sure there are no duplicates in the scansets and that they are sorted.
        for scanset in scansets:
            if not isinstance(scanset, tuple):
                raise ValueError(f"Scanset {scanset} is not a tuple.")
            if len(scanset) == 0:
                raise ValueError("Empty scanset found.")
            if not all(isinstance(table, int) for table in scanset):
                raise ValueError(f"Scanset {scanset} contains non-integer table IDs.")
            if not all(table > 0 for table in scanset):
                raise ValueError(f"Scanset {scanset} contains non-positive table IDs.")
            if not all(scanset[i] < scanset[i + 1] for i in range(len(scanset) - 1)):
                raise ValueError(f"Scanset {scanset} is not sorted.")
            if len(scanset) != len(set(scanset)):
                raise ValueError(f"Scanset {scanset} contains duplicate table IDs.")

        # Make sure that all tables between min table and max table are present.
        min_table = min(table for scanset in scansets for table in scanset)
        max_table = max(table for scanset in scansets for table in scanset)
        for table in range(min_table, max_table + 1):
            if not any(table in scanset for scanset in scansets):
                raise ValueError(f"Table {table} is missing from all scansets.")

    @staticmethod
    def _validate_input_scansets(scansets):
        if not scansets:
            raise ValueError("Scanset list is empty.")
        for scanset in scansets:
            if not isinstance(scanset, (list, tuple)):
                raise TypeError(
                    f"Invalid scanset type: {type(scanset)}. Expected list or tuple."
                )
            if not all(isinstance(table_id, int) for table_id in scanset):
                raise ValueError(
                    f"Invalid table ID in scanset: {scanset}. All IDs should be integers."
                )

    def get_stats(self):
        return self.stats

    def match_tables(self, redset_scansets, benchmark_scansets, config):
        # Count tables in each set of scanset
        num_redset_tables = ScansetMapper._count_tables(redset_scansets)
        num_benchmark_tables = ScansetMapper._count_tables(benchmark_scansets)

        if num_redset_tables > num_benchmark_tables:
            log(
                f"Warning: The determined schema size of the original RedSet cluster (size: {num_redset_tables}) is larger than the number of tables available in the support database ({num_benchmark_tables}). This indicates that the selected combination of redset cluster and support database might not be suitable for realistic workload generation (e.g. overly pessimistic caching estimates, different read/write patterns since multiple tables mapped, ...). Proceed with caution.",
                log_mode="warning",
            )
        else:
            log(
                f"Original RedSet schema size: {num_redset_tables}, support database schema size: {num_benchmark_tables}",
            )

        # Remove duplicates from redset_scansets and count occurences
        distinct_redset_scansets = list(set(redset_scansets))
        redset_scanset_counters = Counter(redset_scansets)
        c_redset_scanset_counters = [0] * len(distinct_redset_scansets)
        for idx, scanset in enumerate(distinct_redset_scansets):
            c_redset_scanset_counters[idx] = redset_scanset_counters[scanset]

        # Map each table to the indices of the scansets it appears in
        table_to_scansets = [[] for _ in range(num_redset_tables + 1)]
        for idx, scanset in enumerate(distinct_redset_scansets):
            for r_table in scanset:
                table_to_scansets[r_table].append(idx)

        n_threads = config.get("num_threads", os.cpu_count())
        if sum(map(len, distinct_redset_scansets)) < 500:
            n_default_iterations = 8
        elif sum(map(len, distinct_redset_scansets)) < 1000:
            n_default_iterations = 4
        else:
            n_default_iterations = 2

        n_iterations_per_thread = config.get(
            "iterations_per_thread", n_default_iterations
        )

        log(
            f"Starting table matching with {n_threads} threads and {n_iterations_per_thread} iterations per thread."
            + "If this takes too long, consider reducing the number of iterations per thread in the config file."
        )

        log(
            f"Scanset statistics:"
            f"\n  - Unique RedSet scansets: {len(distinct_redset_scansets)} (total occurrences: {sum(c_redset_scanset_counters)})"
            f"\n  - Benchmark scansets: {len(benchmark_scansets)} (total table accesses: {sum(map(len, benchmark_scansets))})"
            f"\n  - RedSet tables: {num_redset_tables}"
            f"\n  - Benchmark tables: {num_benchmark_tables}"
        )

        # Find the optimal mapping from redset tables to benchmark tables
        hillmapper_start = time.time()
        best_distance, best_assignment = hillmapper.find_optimal_bijection(
            n_threads,
            n_iterations_per_thread,
            c_redset_scanset_counters,
            distinct_redset_scansets,
            benchmark_scansets,
            table_to_scansets,
            num_redset_tables,
            num_benchmark_tables,
        )
        hillmapper_end = time.time()

        self.stats = {
            "best_distance": best_distance,
            "redset_scansets_count": len(redset_scansets),
            "distinct_redset_scansets_count": len(distinct_redset_scansets),
            "num_redset_tables": num_redset_tables,
            "average_tables_per_scanset": statistics.fmean(map(len, redset_scansets)),
            "average_tables_per_distinct_scanset": statistics.fmean(
                map(len, distinct_redset_scansets)
            ),
            "average_tables_per_benchmark_scanset": statistics.fmean(
                map(len, benchmark_scansets)
            ),
            "n_iterations_per_thread": n_iterations_per_thread,
            "n_threads": n_threads,
            "hillmapper_time": hillmapper_end - hillmapper_start,
        }

        return best_assignment
