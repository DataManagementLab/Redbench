from functools import lru_cache

import duckdb

from redbench.utils.log import log


class NoDataInClusterError(Exception):
    def __init__(self, message="No data found in the specified cluster."):
        self.message = message
        super().__init__(self.message)


def get_scanset_from_redset_query(query):
    return get_scanset_from_redset_query_raw(query["read_table_ids"])


@lru_cache(maxsize=None)
def get_scanset_from_redset_query_raw(redset_read_table_ids):
    if redset_read_table_ids is None:
        return []
    return tuple(sorted(map(int, redset_read_table_ids.split(","))))


# register query hash udf
def query_hash(feature_fingerprint, num_scans, num_joins, read_table_ids):
    hash_str = f"{feature_fingerprint}#{num_scans}#{num_joins}"

    if read_table_ids is not None:  # can be null for write-only queries
        scanset = get_scanset_from_redset_query_raw(read_table_ids)
        hash_str += f"#{scanset}"
    return hash_str


def load_and_preprocess_redset(
    start_date: str,
    end_date: str,
    database_id: int = None,
    instance_id: int = None,
    redset_path: str = None,
    include_copy: bool = False,
    include_analyze: bool = False,
    include_ctas: bool = False,
    include_all_qtypes: bool = False,
    exclude_tables_never_read: bool = False,
    load_full_redset_and_cache: bool = False,
    con: duckdb.DuckDBPyConnection = None,
    limit_rows: int = None,
) -> duckdb.DuckDBPyConnection:
    where_conditions = []
    where_conditions_postprocess = []
    where_conditions_after_cache_restore = []

    # assemble filter conditions to prune away data from the redset the user is not interested in
    tmp_conds = [
        "was_aborted = 0",
    ]

    if start_date is not None and end_date is not None:
        tmp_conds.append(
            f"arrival_timestamp BETWEEN TIMESTAMP '{start_date}' AND TIMESTAMP '{end_date}'"
        )

    if database_id is not None:
        tmp_conds.insert(0, f"database_id = {database_id}")
    if instance_id is not None:
        tmp_conds.insert(1, f"instance_id = {instance_id}")

    if not load_full_redset_and_cache:
        where_conditions += tmp_conds
    else:
        where_conditions_postprocess += tmp_conds

    # query types to include in the sampling
    query_types = ["select", "insert", "delete", "update"]
    if include_copy:
        query_types.append("copy")
    if include_analyze:
        query_types.append("analyze")
    if include_ctas:
        query_types.append("ctas")
    if include_all_qtypes:
        query_types.extend(
            ["other", "analyze", "unload", "vacuum", "copy", "ctas", "analyze"]
        )
        query_types = list(set(query_types))  # deduplicate

    # query_type and other information are missing for result-cached queries. Hence apply conditions after restoring the original values of cached queries
    where_conditions.append(f"(was_cached=1 OR query_type IN {tuple(query_types)})")
    where_conditions_after_cache_restore.append(
        f"query_type IN {tuple(query_types)}"
    )  # use tuple to ensure correct SQL syntax

    # mbytes scanned must not be null
    where_conditions.append("(was_cached=1 OR mbytes_scanned IS NOT NULL)")
    where_conditions_after_cache_restore.append(
        "mbytes_scanned IS NOT NULL"
    )  # this affects only 3 queries in the serverless dataset

    # read and write table ids must not be null
    where_conditions.append(
        "(was_cached=1 OR query_type NOT IN ('select','analyze','ctas', 'update') OR read_table_ids IS NOT NULL)"
    )
    where_conditions_after_cache_restore.append(
        "(query_type NOT IN ('select','analyze','ctas', 'update') OR read_table_ids IS NOT NULL)"
    )
    where_conditions.append(
        "(was_cached=1 OR query_type NOT IN ('insert','update','delete','ctas') OR write_table_ids IS NOT NULL)"
    )
    where_conditions_after_cache_restore.append(
        "(query_type NOT IN ('insert','update','delete','ctas') OR write_table_ids IS NOT NULL)"
    )

    # join the conditions to create the WHERE clause
    load_where_clause = " AND ".join(where_conditions)
    postprocess_where_clause = " AND ".join(where_conditions_postprocess)

    if con is None:
        # read parquet dataset
        con = duckdb.connect(database=":memory:")

        # load the parquet file - prune away query info we are not interested in
        load_query = f"CREATE TABLE redset AS FROM read_parquet('{redset_path}') WHERE {load_where_clause};"
        con.execute(load_query)

    # retrieve redset columns
    redset_columns = con.query("SELECT * FROM redset LIMIT 1").df().columns.tolist()
    redset_columns_renamed = [
        f"o.{col}"
        if col not in ["arrival_timestamp", "query_id", "was_cached"]
        else f"c.{col}"
        for col in redset_columns
    ]

    # Register the query_hash function only if it does not already exist
    try:
        con.create_function("query_hash", query_hash, [str, int, int, str], str)
    except Exception as e:
        # Function already exists, ignore the error
        log(e)
        pass

    # preprocess redset
    # map (result) cached query instances to their original query instance
    # this is done to ensure that the workload contains all query instances, including the cached ones
    # this is important to accurately represent the query repetitions in the workload
    # the cached query instances are mapped to their original query instance by joining on the cache_source_query_id
    # and the instance_id, database_id, and user_id
    if len(postprocess_where_clause) == 0:
        redset_filtered_sql = "SELECT * FROM redset"
    else:
        redset_filtered_sql = f"SELECT * FROM redset WHERE {postprocess_where_clause}"

    preprocess_query = f"""
    DROP TABLE IF EXISTS redset_preprocessed;
    CREATE TABLE redset_preprocessed AS
    WITH redset_filtered AS (
        {redset_filtered_sql}
    ),
    original AS (
        SELECT * FROM redset_filtered WHERE was_cached = 0
    ),
    cached AS (
        SELECT
            {", ".join(redset_columns_renamed)}
        FROM redset_filtered c
        JOIN original o
            ON c.cache_source_query_id = o.query_id AND c.instance_id = o.instance_id AND c.database_id = o.database_id AND c.user_id = o.user_id
        WHERE c.was_cached = 1
    ), combined AS (
        SELECT * FROM original
        UNION ALL
        SELECT * FROM cached
    )
    """

    if exclude_tables_never_read:
        # 1. get all tables that are read from
        # 2. get all queries that write to one of these tables (and do not read) + get all queries are not writing or read-only
        preprocess_query += """
        , read_tables AS (
            SELECT DISTINCT
                instance_id,
                database_id,
                unnest(string_split(read_table_ids, ',')) AS table_id
            FROM combined
            WHERE read_table_ids IS NOT NULL
        ), table_filtered AS (
            (
                SELECT c.* FROM combined c, read_tables rt
                WHERE c.write_table_ids = rt.table_id
                AND c.instance_id = rt.instance_id
                AND c.database_id = rt.database_id
                AND c.read_table_ids IS NULL
                AND c.write_table_ids IS NOT NULL
            ) UNION (
                SELECT * FROM combined c
                WHERE c.write_table_ids IS NULL OR c.read_table_ids IS NOT NULL
            )
        )
        """
    else:
        preprocess_query += ", table_filtered AS (SELECT * FROM combined) "

    if len(where_conditions_after_cache_restore) > 0:
        final_where_clause = " AND ".join(where_conditions_after_cache_restore)
        final_where_clause = f"WHERE {final_where_clause}"
    else:
        final_where_clause = ""

    if limit_rows is not None:
        limit_str = f" LIMIT {limit_rows} "
    else:
        limit_str = ""

    preprocess_query += f"SELECT *, query_hash(feature_fingerprint, num_scans, num_joins, read_table_ids) as query_hash FROM table_filtered {final_where_clause} ORDER BY arrival_timestamp ASC {limit_str};"
    # print(preprocess_query)
    con.execute(preprocess_query)

    # if database_id is not None and instance_id is not None:
    #     # Check if the database_id and instance_id exist
    #     query = f"SELECT COUNT(*) FROM redset WHERE database_id = {database_id} AND instance_id = {instance_id};"
    #     result = con.query(query).df()
    #     if result.iloc[0, 0] == 0:
    #         raise NoDataInClusterError(
    #             f"No data found for database_id {database_id} and instance_id {instance_id} in the dataset for the timeframe {start_date}-{end_date}. Please consider using a different database_id or instance_id,  enable random databases with --enable_random_databases or extend the timeframe."
    #         )

    return con


def determine_redset_dataset_type(redset_path: str):
    # extract if using serverless or provisioned from redset
    if "serverless" in redset_path:
        redset_dataset_type = "serverless"
    elif "provisioned" in redset_path:
        redset_dataset_type = "provisioned"
    else:
        raise ValueError(
            f"Could not determine redset dataset type (serverless or provisioned) from the redset path: {redset_path}"
        )
    return redset_dataset_type
