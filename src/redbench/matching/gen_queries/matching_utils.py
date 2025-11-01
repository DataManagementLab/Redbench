import re
from typing import Dict

from pyparsing import lru_cache


@lru_cache(maxsize=1000)
def load_query(query_filepath):
    with open(query_filepath, "r") as f:
        query = f.read()
    query = query.strip()
    if not query.endswith(";"):
        query += ";"
    return query


def init_query(query_filepath, table_versions: Dict = None, remove_linebreaks=False):
    query = load_query(query_filepath)
    if table_versions is not None:
        for table, version in table_versions.items():
            query = query.replace(table, version)

    if remove_linebreaks:
        query = query.replace("\n", " ")

    return query


class SimpleDmlsConstructor:
    def __init__(self, benchmark_db):
        self.benchmark_db = benchmark_db

    def __call__(self, query):
        # table name is still an integer (maybe also 1_1 with version suffix if active)
        table_name = query["benchmark_write_table"]

        if bool(re.compile(r"_(\d+)$").search(str(table_name))):
            # Remove version suffix if present
            table_name = re.sub(r"_(\d+)$", "", str(table_name))

        col_names = self._get_col_names(table_name)

        # Sample random row from the table
        target_row = self.benchmark_db.execute(
            f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1"
        ).fetchone()

        def escape_sql_value(value):
            if isinstance(value, str):
                # Escape single quotes with '' in SQL
                return value.replace("'", "''")
            return value

        values = ", ".join(
            [
                f"'{escape_sql_value(value)}'" if value is not None else "NULL"
                for value in target_row
            ]
        )
        target_row = [  # in case of NULL use IS NULL instead of = NULL (evaluates to unknown)
            f"= '{escape_sql_value(value)}'" if value is not None else "IS NULL"
            for value in target_row
        ]

        # Create 2 DMLs: one to delete the row and one to insert it back
        delete_query = f"""
            DELETE FROM {table_name}
            WHERE {" AND ".join([f"{col} {value}" for col, value in zip(col_names, target_row)])}
        """.strip().replace("\n", " ")
        insert_query = f"""
            INSERT INTO {table_name} ({", ".join(col_names)})
            VALUES ({values})
        """.strip().replace("\n", " ")

        # Concatenate and return the 2 DML queries
        return f"{delete_query}; {insert_query};"

    @lru_cache(maxsize=1000)
    def _get_col_names(self, table_name: str):
        return (
            self.benchmark_db.execute(f"PRAGMA table_info({table_name})")
            .df()["name"]
            .tolist()
        )
