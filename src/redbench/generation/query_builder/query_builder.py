import re

import numpy as np
import pandas as pd

from redbench.generation.helper.workload_statistics_retriever import (
    DatabaseStatisticsRetriever,
)
from redbench.generation.query_builder.column_type_retriever import (
    ColumnType,
    retrieve_column_type,
)
from redbench.generation.query_builder.join_clause_builder import (
    build_join_clauses,
    build_join_conditions,
)
from redbench.generation.query_builder.predicate_builder import build_predicate


def remove_duplicated_white_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def build_select_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    min_filter_selectivity: float,
    aggregation="*",
    add_distinct=True,
    override_predicate=None,
    simple_agg=False,
):
    # Extract required parameters
    start_t = query.get("start_t")  # Main table
    join_tables = query.get("join_tables", set())  # Set of tables to join

    # Ensure start_t is valid and in join_tables
    if not start_t:
        raise ValueError("start_t (main table) must be specified!")
    if not join_tables:
        raise ValueError("join_tables cannot be empty!")
    if start_t not in join_tables:
        raise ValueError("start_t must be included in join_tables!")

    # Build SELECT clause

    table_info = database_knowledge.retrieve_table_info(start_t)
    if not table_info:
        raise ValueError(f"Table {start_t} not found in schema!")

    pk_columns = [col for col, props in table_info.items() if props.get("pk", False)]
    columns = [col for col, props in table_info.items()]

    if simple_agg:
        chosen_columns = rand_state.choice(
            columns, size=max(1, min(query["num_aggregations"], len(columns)))
        )
        aggregation = ", ".join([f'"{start_t}"."{c}"' for c in chosen_columns])

    pk_columns = ", ".join([f'"{start_t}"."{c}"' for c in pk_columns])
    distinct_query = f"DISTINCT ON({pk_columns})"
    select_clause = f"SELECT {distinct_query} {aggregation}"

    # Build FROM clause using start_t
    from_clause = f'FROM "{start_t}"'

    # Exclude start_t from joins since it's already the main table
    # other_tables = join_tables - {start_t}

    # Build JOIN conditions
    join_condition_clauses = build_join_clauses(query)
    join_clause = " ".join(join_condition_clauses)
    # Build WHERE conditions
    if override_predicate:
        predicate_clauses, approximated_selectivities = override_predicate
    else:
        predicate_clauses, approximated_selectivities = build_predicate(
            query,
            database_knowledge,
            rand_state,
            min_filter_selectivity=min_filter_selectivity,
        )
    where_clause = (
        f"WHERE {' AND '.join(predicate_clauses)}" if predicate_clauses else ""
    )

    # Construct final query
    sql_query = f"{select_clause} {from_clause} {join_clause} {where_clause};"

    return (
        remove_duplicated_white_spaces(sql_query.strip()),
        remove_duplicated_white_spaces(sql_query.strip()),
        approximated_selectivities,
    )


def build_delete_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    min_filter_selectivity: float,
) -> tuple[str, str]:
    start_t = query.get("start_t")
    join_tables = query.get("join_tables", set())

    if not start_t:
        raise ValueError("start_t (main table) must be specified!")
    if not join_tables:
        raise ValueError("join_tables cannot be empty!")
    if start_t not in join_tables:
        raise ValueError("start_t must be included in join_tables!")

    table_info = database_knowledge.retrieve_table_info(start_t)
    if not table_info:
        raise ValueError(f"Table {start_t} not found in schema!")
    other_tables = join_tables - {start_t}
    join_condition_clauses = build_join_conditions(query)
    table_joins = [f'"{t}"' for t in other_tables]
    using_clause = f"USING {', '.join(table_joins)}" if other_tables else ""
    where_conditions, approximated_selectivities = build_predicate(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity=min_filter_selectivity,
    )
    join_conditions = " AND ".join(join_condition_clauses)

    where_clause = (
        "WHERE " + " AND ".join(filter(None, [join_conditions] + where_conditions))
        if join_conditions or where_conditions
        else ""
    )

    sql_query = f'DELETE FROM "{start_t}" {using_clause} {where_clause};'

    table_columns = table_info.keys()
    select_columns = ", ".join([f'"{start_t}"."{c}"' for c in table_columns])

    # Generate the SELECT query
    select_query, _, approximated_scan_selectivities = build_select_query(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity,
        select_columns,
        True,
        override_predicate=(where_conditions, approximated_selectivities),
    )
    return (
        remove_duplicated_white_spaces(sql_query.strip()),
        select_query,
        approximated_scan_selectivities,
    )


def get_random_varchar(
    rand_state: np.random.RandomState,
    column_name: str,
    database_knowledge: DatabaseStatisticsRetriever,
    table_name: str,
):
    max_length = (
        database_knowledge.retrieve_varchar_lengths()
        .get(table_name, {})
        .get(column_name, 255)
    )

    length = rand_state.randint(1, max_length + 1)
    random_string = "".join(
        rand_state.choice(
            list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            size=length,
        )
    )
    return f"'{random_string}'"


def build_update_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    min_filter_selectivity: float,
) -> tuple[str, str]:
    start_t = query.get("start_t")
    if not start_t:
        raise ValueError("start_t (main table) must be specified!")

    # Extract column info from schema
    table_info = database_knowledge.retrieve_table_info(start_t)
    if not table_info:
        raise ValueError(f"Table {start_t} not found in schema!")

    # Extract non-primary key columns
    non_pk_columns = [
        col for col, props in table_info.items() if not props.get("pk", False)
    ]
    if not non_pk_columns:
        raise ValueError(f"No non-primary key columns found for table {start_t}")

    # Choose a random non-primary key column
    column = rand_state.choice(non_pk_columns)
    column_type = retrieve_column_type(table_info[column]["type"])

    if column_type == ColumnType.VARCHAR:
        new_value = get_random_varchar(rand_state, column, database_knowledge, start_t)
    elif column_type == ColumnType.INT:
        new_value = str(rand_state.randint(1, 100))
    elif column_type == ColumnType.FLOAT:
        new_value = str(round(rand_state.uniform(1, 100), 2))
    elif column_type == ColumnType.DATE:
        new_value = f"'2024-01-{rand_state.randint(1, 28)}'"
    else:
        new_value = "NULL"

    join_tables = query.get("join_tables", set()) - {start_t}
    join_condition_clauses = build_join_conditions(query)

    table_joins = [f'"{t}"' for t in join_tables]
    from_clause = f"FROM {', '.join(table_joins)}" if join_tables else ""
    join_conditions = " AND ".join(join_condition_clauses)
    where_conditions, approximated_selectivities = build_predicate(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity=min_filter_selectivity,
    )

    where_clause = (
        "WHERE " + " AND ".join(filter(None, [join_conditions] + where_conditions))
        if join_conditions or where_conditions
        else ""
    )

    sql_query = (
        f'UPDATE "{start_t}" SET "{column}" = {new_value} {from_clause} {where_clause};'
    )

    table_columns = table_info.keys()
    select_columns = ", ".join([f'"{start_t}"."{c}"' for c in table_columns])

    # Generate the SELECT query
    select_query, _, approximated_selectivities = build_select_query(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity,
        select_columns,
        True,
        override_predicate=(where_conditions, approximated_selectivities),
    )
    return (
        remove_duplicated_white_spaces(sql_query.strip()),
        select_query,
        approximated_selectivities,
    )


def build_insert_select_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    add_conflict_logic=False,
    min_filter_selectivity: float = None,
) -> tuple[str, str]:
    write_table = query.get("write_table")
    start_t = query.get("start_t")

    if not write_table:
        raise ValueError("write_table (target table) must be specified!")
    if not start_t:
        raise ValueError("start_t (source table) must be specified!")

    # Ensure write_table exists in the schema
    table_info = database_knowledge.retrieve_table_info(start_t)
    pk_columns = [col for col, props in table_info.items() if props.get("pk", False)]
    # Get all column names from write_table
    table_columns = table_info.keys()
    select_columns = ", ".join([f'"{start_t}"."{c}"' for c in table_columns])

    write_columns = ", ".join([f'"{c}"' for c in table_columns])
    raw_pk_columns = ", ".join([f'"{c}"' for c in pk_columns])

    # Generate the SELECT query
    select_query, _, approximated_selectivities = build_select_query(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity,
        select_columns,
        True,
    )
    on_conflict_part = (
        f"ON CONFLICT ({raw_pk_columns}) DO NOTHING" if add_conflict_logic else ""
    )
    # Construct the INSERT INTO ... SELECT query
    sql_query = f'INSERT INTO "{write_table}" ({write_columns}) {select_query[:-1]} {on_conflict_part};'

    return (
        remove_duplicated_white_spaces(sql_query.strip()),
        select_query,
        approximated_selectivities,
    )


def build_ctas_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    min_filter_selectivity: float,
) -> tuple[str, str]:
    """
    Builds a CREATE TABLE AS (CTAS) query to create a new table and populate it using a SELECT query.
    """
    write_table = query.get("write_table")
    start_t = query.get("start_t")

    if not write_table:
        raise ValueError("write_table (target table) must be specified!")
    if not start_t:
        raise ValueError("start_t (source table) must be specified!")

    table_info = database_knowledge.retrieve_table_info(start_t)
    pk_columns_start = [
        f'"{col}"' for col, props in table_info.items() if props.get("pk", False)
    ]
    pk_columns_raw = ", ".join([f'"{c}"' for c in pk_columns_start])

    # Get all column names from start_t
    table_columns = table_info.keys()
    column_list = ", ".join([f'"{start_t}"."{c}"' for c in table_columns])

    # Generate the SELECT query
    select_query, _, approximated_selectivities = build_select_query(
        query,
        database_knowledge,
        rand_state,
        min_filter_selectivity,
        column_list,
        True,
    )

    # Construct the CTAS query
    sql_query = f"CREATE TABLE {write_table} AS {select_query}; ALTER TABLE {write_table} ADD PRIMARY KEY ({pk_columns_raw});"

    return (
        remove_duplicated_white_spaces(sql_query.strip()),
        select_query,
        approximated_selectivities,
    )


def build_copy_query(
    query: pd.Series,
    database_knowledge: DatabaseStatisticsRetriever,
    rand_state: np.random.RandomState,
    max_split_id: int,
):
    write_table = query.get("write_table")

    if not write_table:
        raise ValueError("write_table (target table) must be specified!")

    table_info = database_knowledge.retrieve_table_info(write_table)

    # Get all column names from write_table
    table_columns = table_info.keys()
    table_n = write_table.rsplit("_", 1)[0]

    # WE HAVE NO GUARANTEES THAT THIS FILE IS LOADED ONY ONCE! I.E. PRIMARY-KEY CONSTRAINTS ARE NOT GUARANTEED!
    path = f"<<csv_path_placeholder>>/{table_n}_{max_split_id}.csv"

    write_columns = ", ".join(table_columns)
    sql_query = f"COPY {write_table}({write_columns}) FROM '{path}' WITH (FORMAT csv, HEADER true, DELIMITER ',', NULL '<!NULL-?>');"
    return remove_duplicated_white_spaces(sql_query), None, []


# if __name__ == "__main__":
#     # Define a sample row for the DataFrame
#     query_data = {
#         "query_type": "SELECT",
#         "num_joins": 2,
#         "start_t": "managershalf_0",
#         "joins_t": [('managershalf_0', ['managerID'], 'managers_0', ['managerID'], True)],
#         "join_tables": {'managershalf_0', 'managers_0'},
#         "write_table": 'managershalf_1',
#         "join_tables_with_selectivity": {
#             'managershalf_0': 0.25578288319273046,
#             'managers_0': 0.6246364098840379
#         }
#     }

#     # Convert to a DataFrame row
#     query_df = pd.DataFrame([query_data])

#     database_knowledge = DatabaseStatisticsRetriever(2)
#     log(query_df.iloc[0])
#     randstate = np.random.RandomState()
#     log(build_select_query(query_df.iloc[0], database_knowledge, randstate))
#     log(build_delete_query(query_df.iloc[0], database_knowledge, randstate))
#     log(build_update_query(query_df.iloc[0], database_knowledge, randstate))
#     log(build_insert_select_query(query_df.iloc[0], database_knowledge, randstate))
