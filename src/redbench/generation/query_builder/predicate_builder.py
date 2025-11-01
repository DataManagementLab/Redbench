from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from redbench.generation.dataset_input.prepare_and_scale import extract_scale_columns
from redbench.generation.dataset_input.retrieve_statistics import ColumnStats
from redbench.generation.helper.workload_statistics_retriever import (
    DatabaseStatisticsRetriever,
)
from redbench.generation.query_builder.column_type_retriever import (
    ColumnType,
    retrieve_column_type,
)


def sanitize_value(value, column_type):
    """
    Sanitizes the value to ensure it's safe to use in a SQL query.

    :param value: The value to sanitize.
    :param column_type: The column type (VARCHAR, INT, FLOAT, etc.).
    :return: A sanitized value.
    """
    if column_type == ColumnType.VARCHAR:
        # Escape single quotes by doubling them up for SQL safety
        sanitized_value = str(value).replace("'", "''").strip()
        return f"'{sanitized_value}'"
    elif column_type == ColumnType.INT:
        # Ensure the value is a valid number and not something like '12abc'
        return int(value)
    elif column_type == ColumnType.FLOAT:
        # Handle unexpected types (can be adjusted as needed)
        return float(value)
    elif column_type == ColumnType.DATE:
        # For dates, assume it's already in a suitable format or convert
        return f"'{str(value)}'"
    else:
        # For UNKNOWN or other types, if it's a string, quote it
        if isinstance(value, str):
            sanitized_value = value.replace("'", "''").strip()
            return f"'{sanitized_value}'"
        else:
            return value


def rescale_sigma(sigma: float, min_filter_selectivity: float) -> float:
    """
    Rescales sigma to ensure the minimum selectivity is respected.
    :param sigma: Original sigma value between 0 and 1.
    :param min_filter_selectivity: Minimum selectivity threshold between 0 and 1.
    :return: Rescaled sigma value.
    """
    assert 0 <= sigma <= 1, f"Sigma must be between 0 and 1, but got {sigma}"
    assert 0 <= min_filter_selectivity <= 1, (
        f"Min filter selectivity must be between 0 and 1, but got {min_filter_selectivity}"
    )

    return min_filter_selectivity + (1 - min_filter_selectivity) * sigma


def build_predicate(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    min_filter_selectivity: float,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """
    Generates a list of predicates based on query selectivity and column statistics.
    The range size is fixed by sigma, while the start position is randomized.

    :param query: A Pandas Series containing query information.
    :param column_statistics: A dictionary mapping table.column to quantile statistics.
    :param rand_state: A numpy RandomState object for controlled randomness.
    :return: A list of SQL predicates.
    """
    predicates = []
    approximated_selectivities = []

    start_t = query.get("start_t")
    if not start_t:
        raise ValueError("start_t (main table) must be specified!")

    for table, sigma in query["join_tables_with_selectivity"].items():
        # Rescale sigma to respect minimum selectivity
        if min_filter_selectivity is not None:
            sigma_scaled = rescale_sigma(sigma, min_filter_selectivity)
        else:
            sigma_scaled = sigma

        column_stats: Dict[str, ColumnStats] = (
            database_knowledge.retrieve_column_statistics(table).columns
        )

        # Extract column info from schema
        table_info = database_knowledge.retrieve_table_info(table)

        if not table_info:
            raise ValueError(f"Table {table} not found in schema!")
        items = list(column_stats.items())

        # randomize order of columns to avoid bias
        rand_state.shuffle(items)

        # remove columns which are pk/fk
        valid_columns = extract_scale_columns(
            {
                "relationships": database_knowledge.retrieve_relationships(),
                "table_col_info": database_knowledge.retrieve_table_info(),
            }
        )[table]
        filtered_items = [
            (column, values) for column, values in items if column not in valid_columns
        ]

        for column, values in filtered_items:
            quantiles = values.quantiles
            if None in quantiles:
                continue

            assert column in table_info, (
                f"Column {column} not found in table {table} schema! \n {table_info} \n{column_stats}"
            )
            column_type = retrieve_column_type(table_info[column]["type"])

            # Determine the exact range size based on sigma
            range_size = max(
                0, min(100, int(sigma_scaled * 100))
            )  # Ensure within 1-100

            # Pick a random valid start index so the range fits within bounds
            start_index = rand_state.randint(0, 100 - range_size + 1)

            lower_bound = quantiles[start_index]
            upper_bound = quantiles[start_index + range_size]

            # approximate selectivity
            start_index = quantiles.index(
                lower_bound
            )  # find first index of lower_bound
            end_index = (
                len(quantiles) - 1 - quantiles[::-1].index(upper_bound)
            )  # find last index of upper_bound
            selectivity = (end_index - start_index + 1) / len(
                quantiles
            )  # calculate selectivity
            approximated_selectivities.append((table, column, selectivity))

            # Format bounds based on column type
            lower_bound = sanitize_value(lower_bound, column_type)
            upper_bound = sanitize_value(upper_bound, column_type)

            # Construct SQL predicate
            predicate = f'"{table}"."{column}" BETWEEN {lower_bound} AND {upper_bound}'
            predicates.append(predicate)
            break  # Only pick one column per table

    return predicates, approximated_selectivities


# if __name__ == "__main__":

#     # Define a sample row for the DataFrame
#     query_data = {
#         "query_type": "SELECT",
#         "num_joins": 2,
#         "start_t": "managershalf_0",
#         "joins_t": [('managershalf_0', ['managerID'], 'managers_0', ['managerID'], True)],
#         "join_tables": {'managershalf_0', 'managers_0'},
#         "write_table": None,
#         "join_tables_with_selectivity": {
#             'managershalf_0': 0.25578288319273046,
#             'managers_0': 0.6246364098840379
#         }
#     }

#     # Convert to a DataFrame row

#     query_df = pd.DataFrame([query_data])
#     database_knowledge = DatabaseStatisticsRetriever(2)
#     randstate = np.random.RandomState()

#     log(build_predicate(query_df.iloc[0], database_knowledge, randstate)[0])
