import os

import duckdb
from tqdm import tqdm

from redbench.generation.dataset_input.load_schema import (
    get_sql_schema,
)
from redbench.utils.log import log


def get_db_name_from_schema(schema_file_path):
    """Extracts the database name from the schema file name."""
    return os.path.splitext(os.path.basename(schema_file_path))[0]


def load_schema(conn, schema):
    conn.execute(schema)


def load_csv_files(conn, csv_directory_path):
    """Loads each CSV file into a corresponding table with the same name."""
    for csv_file in tqdm(
        os.listdir(csv_directory_path), desc="Loading CSV files into database"
    ):
        if csv_file.endswith(".csv"):
            table_name = os.path.splitext(csv_file)[0]
            full_path = os.path.join(csv_directory_path, csv_file)

            copy_query = f"COPY \"{table_name}\" FROM '{full_path}' NULL '<!NULL-?>' CSV HEADER QUOTE '\"' ESCAPE '\"'"
            conn.execute(copy_query)


def create_duckdb(
    csv_directory_path: str,
    output_db_path: str,
    sql_schema_path: str,
    force: bool = False,
):
    schema = get_sql_schema(sql_schema_path)
    duckdb_path = os.path.join(output_db_path)

    if not force and os.path.exists(duckdb_path):
        log(f"Database already exists at: {duckdb_path}. Skipping creation.")
        return

    log(f"Creating database at: {duckdb_path}")
    conn = duckdb.connect(duckdb_path)

    load_schema(conn, schema)
    load_csv_files(conn, csv_directory_path)

    conn.commit()
    conn.close()
    log("Database setup complete.")
