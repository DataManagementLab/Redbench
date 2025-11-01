import json
import re
from typing import Dict, Optional

from redbench.generation.dataset_input.load_schema import (
    get_json_schema,
    get_sql_schema,
)
from redbench.generation.dataset_input.retrieve_statistics import (
    ColumnStats,
    TableStats,
)
from redbench.generation.helper.redset_table_sizes import (
    define_sizes_for_redset_tables,
)
from redbench.generation.helper.table_mapper import (
    map_redset_table_to_physical_table_by_table_sizes,
)


def load_database_stats(json_path: str) -> Dict[str, TableStats]:
    with open(json_path, "r") as file:
        data = json.load(file)

    # Convert JSON structure into TableStats and ColumnStats objects
    return {
        table_name: TableStats(
            total_rows=table_data["total_rows"],
            columns={
                col_name: ColumnStats(**col_data)
                for col_name, col_data in table_data["columns"].items()
            },
        )
        for table_name, table_data in data.items()
    }


def modify_json(json_data, words_to_change, addition_str):
    """
    Recursively modifies JSON-like data (dicts and lists) by appending addition_str to exact string matches.

    :param json_data: The input JSON-like structure (dict, list, or primitive values).
    :param words_to_change: A set or list of words to modify.
    :param addition_str: The string to append to matching values.
    :return: A modified JSON structure.
    """
    if isinstance(json_data, dict):
        return {
            key: modify_json(value, words_to_change, addition_str)
            for key, value in json_data.items()
        }
    elif isinstance(json_data, list):
        return [modify_json(item, words_to_change, addition_str) for item in json_data]
    elif isinstance(json_data, str) and json_data in words_to_change:
        return json_data + addition_str
    else:
        return json_data


def modify_dict_keys(
    original_dict, words_to_change, addition_str, root_only: bool = False
):
    """
    Recursively modifies dictionary keys by appending addition_str to exact matches.

    :param original_dict: The original dictionary.
    :param words_to_change: A set or list of keys to modify.
    :param addition_str: The string to append to matching keys.
    :return: A new dictionary with modified keys.
    """
    if not isinstance(original_dict, dict):
        return original_dict  # Base case: return non-dict values as-is

    modified_dict = {}
    for key, value in original_dict.items():
        # Append addition_str to the key if it matches any in words_to_change
        new_key = f"{key}{addition_str}" if key in words_to_change else key

        if root_only:
            modified_dict[new_key] = value
        else:
            modified_dict[new_key] = modify_dict_keys(
                value, words_to_change, addition_str
            )  # Recurse for nested dicts

    return modified_dict


class DatabaseStatisticsRetriever:
    def __init__(
        self, num_universes, column_statistics_path, json_schema_path, sql_schema_path
    ):
        self.num_universes = num_universes
        self.column_statistics_path = column_statistics_path
        self.json_schema_path = json_schema_path
        self.sql_schema_path = sql_schema_path

        self._table_names = None
        self._column_statistic = None
        self._relationships = None
        self._table_column_info = None
        self._mapper = None
        self._varchar_lengths = self.retrieve_varchar_lengths()

    def get_default_table_names(self):
        if self._table_names:
            return self._table_names

        data = load_database_stats(self.column_statistics_path)
        table_names = list(data.keys())
        self._table_names = table_names
        return self._table_names

    def get_all_table_names(self):
        return list(set(self._mapper.values()))

    def get_original_table_names(self):
        table_names = self.get_default_table_names()
        result = []
        for multiverse_id in range(self.num_universes):
            add_str = f"_{multiverse_id}"
            result.extend([t + add_str for t in table_names])
        return result

    def retrieve_column_statistics(self, table_name=None):
        if self._column_statistic:
            return (
                self._column_statistic[table_name.split("_ctasc2b89z8c2z9_")[0]]
                if table_name
                else self._column_statistic
            )
        data = load_database_stats(self.column_statistics_path)
        table_names = list(data.keys())
        result_dict = {}
        for multiverse_id in range(self.num_universes):
            add_str = f"_{multiverse_id}"
            universe_dict = modify_dict_keys(data, table_names, add_str, root_only=True)

            result_dict = result_dict | universe_dict
        self._column_statistic = result_dict
        return (
            self._column_statistic[table_name.split("_ctasc2b89z8c2z9_")[0]]
            if table_name
            else self._column_statistic
        )

    def retrieve_relationships(self):
        if self._relationships:
            return self._relationships
        schema = get_json_schema(self.json_schema_path)

        result_list = []
        for multiverse_id in range(self.num_universes):
            add_str = f"_{multiverse_id}"
            universe_list = modify_json(
                schema["relationships"], self.get_default_table_names(), add_str
            )

            result_list.extend(universe_list)
        self._relationships = result_list
        return self._relationships

    def add_new_relation(self, relation):
        if self._relationships:
            self._relationships.append(relation)

    def retrieve_table_info(self, table_name=None):
        if self._table_column_info:
            return (
                self._table_column_info[table_name.split("_ctasc2b89z8c2z9_")[0]]
                if table_name
                else self._table_column_info
            )

        schema = get_json_schema(self.json_schema_path)

        result_dict = {}
        for multiverse_id in range(self.num_universes):
            add_str = f"_{multiverse_id}"
            universe_dict = modify_dict_keys(
                schema["table_col_info"],
                self.get_default_table_names(),
                add_str,
                root_only=True,
            )

            result_dict = result_dict | universe_dict
        self._table_column_info = result_dict
        return (
            self._table_column_info[table_name.split("_ctasc2b89z8c2z9_")[0]]
            if table_name
            else self._table_column_info
        )

    def compute_mapping(self, sampled_groups):
        # compute mapping from redset tables to physical tables: map largest redset tables to largest physical tables
        phys_table_stats = self.retrieve_column_statistics()
        redset_table_sizes_dict, redset_all_table_ids, write_all_table_ids = (
            define_sizes_for_redset_tables(sampled_groups)
        )
        self._mapper = map_redset_table_to_physical_table_by_table_sizes(
            redset_table_sizes_dict,
            phys_table_stats,
        )

        # map remaining tables (where we had not enough statistics) randomly
        phys_table_names = list(phys_table_stats.keys())
        for redset_table_id in redset_all_table_ids.union(write_all_table_ids):
            if redset_table_id not in self._mapper:
                idx = redset_table_id % len(phys_table_names)
                self._mapper[redset_table_id] = phys_table_names[idx]

    def retrieve_mapping(self):
        return self._mapper

    def update_mapping(self, old_redset_table, table_name):
        self._mapper[old_redset_table] = table_name

    def retrieve_varchar_lengths(self):
        sql_schema = get_sql_schema(self.sql_schema_path, keep_newline=True)
        table_pattern = re.compile(
            r"CREATE TABLE\s+(?:IF NOT EXISTS\s+)?(?P<name>(?:\"[^\"]+\"|\w+)(?:\.(?:\"[^\"]+\"|\w+))?)[\s\r\n]*\((?P<body>.*?)\);",
            re.IGNORECASE | re.DOTALL,
        )
        column_pattern = re.compile(
            r'(?P<column>"[^"]+"|\b\w+\b)\s+(?:character varying|varchar|char)\s*(?:\(\s*(?P<length>\d+)\s*\))?',
            re.IGNORECASE,
        )

        raw_varchar_lengths: Dict[str, Dict[str, Optional[int]]] = {}
        for table_match in table_pattern.finditer(sql_schema):
            raw_table_name = table_match.group("name")
            table_body = table_match.group("body")

            # Extract final identifier and strip surrounding quotes.
            if "." in raw_table_name:
                raw_table_name = raw_table_name.split(".")[-1]
            table_name = raw_table_name.strip('"')

            table_columns: Dict[str, Optional[int]] = {}
            for column_match in column_pattern.finditer(table_body):
                column_identifier = column_match.group("column")
                column_name = (
                    column_identifier[1:-1]
                    if column_identifier.startswith('"')
                    and column_identifier.endswith('"')
                    else column_identifier
                )
                length_str = column_match.group("length")
                length = int(length_str) if length_str is not None else None
                table_columns[column_name] = length

            raw_varchar_lengths[table_name] = table_columns

        aggregated_lengths: Dict[str, Dict[str, Optional[int]]] = {}
        for table_name, columns in raw_varchar_lengths.items():
            normalized_name = table_name
            target_columns = aggregated_lengths.setdefault(normalized_name, {})
            for column_name, length in columns.items():
                if column_name not in target_columns:
                    target_columns[column_name] = length
                    continue

                existing_length = target_columns[column_name]
                if existing_length is None or length is None:
                    target_columns[column_name] = None
                elif existing_length != length:
                    target_columns[column_name] = max(existing_length, length)

        return aggregated_lengths
