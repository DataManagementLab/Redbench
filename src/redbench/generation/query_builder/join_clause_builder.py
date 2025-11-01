from typing import List

import pandas as pd


def build_join_conditions(query: pd.Series) -> List[str]:
    """
    Generates a list of predicates based on query selectivity and column statistics.
    The range size is fixed by sigma, while the start position is randomized.

    :param query: A Pandas Series containing query information.
    :param column_statistics: A dictionary mapping table.column to quantile statistics.
    :param rand_state: A numpy RandomState object for controlled randomness.
    :return: A list of SQL predicates.
    """
    clauses = []

    for left_table, left_keys, right_table, right_keys, is_inner in query["joins_t"]:
        conditions = " AND ".join(
            f'"{left_table}"."{lk}" = "{right_table}"."{rk}"'
            for lk, rk in zip(left_keys, right_keys)
        )
        clauses.append(conditions)

    return clauses


def build_join_clauses(query: pd.Series) -> List[str]:
    """
    Generates SQL JOIN clauses based on query join conditions.

    :param query: A Pandas Series containing query information with a "joins_t" key.
    :return: A list of SQL JOIN clauses.
    """
    clauses = []

    for left_table, left_keys, right_table, right_keys, is_inner in query["joins_t"]:
        if is_inner:
            join_type = "JOIN"
        else:
            join_type = "LEFT JOIN"
        conditions = " AND ".join(
            f'"{left_table}"."{lk}" = "{right_table}"."{rk}"'
            for lk, rk in zip(left_keys, right_keys)
        )
        clauses.append(f"{join_type} {right_table} ON {conditions}")

    return clauses


# if __name__ == "__main__":
#     # Define a sample row for the DataFrame with multiple joins
#     query_data = {
#         "query_type": "SELECT",
#         "num_joins": 3,
#         "start_t": "employees",
#         "joins_t": [
#             ('employees', ['employeeID'], 'departments', ['departmentID'], True),   # employees -> departments (INNER JOIN)
#             ('departments', ['managerID'], 'managers', ['managerID'], False),       # departments -> managers (LEFT JOIN)
#             ('managers', ['regionID'], 'regions', ['regionID'], True)               # managers -> regions (INNER JOIN)
#         ],
#         "join_tables": {'employees', 'departments', 'managers', 'regions'},
#         "write_table": None,
#         "join_tables_with_selectivity": {
#             'employees': 0.5,
#             'departments': 0.3,
#             'managers': 0.6,
#             'regions': 0.4
#         }
#     }

#     # Convert to a DataFrame row
#     query_df = pd.DataFrame([query_data])


#     log(query_df.iloc[0])

#     log(build_join_conditions(query_df.iloc[0]))
#     log(build_join_clauses(query_df.iloc[0]))
