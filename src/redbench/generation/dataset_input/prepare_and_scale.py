import collections
import functools
import json
import multiprocessing
import os
import re
from os.path import join
from typing import Dict

from tqdm import tqdm

from redbench.generation.dataset_input.load_schema import (
    get_json_schema,
    get_sql_schema,
)
from redbench.generation.dataset_input.read_csv import read_csv
from redbench.generation.dataset_input.retrieve_statistics import TableStats
from redbench.generation.helper.workload_statistics_retriever import (
    load_database_stats,
)
from redbench.generation.query_builder.column_type_retriever import (
    ColumnType,
    retrieve_column_type,
)
from redbench.utils.log import log


def scale_file(
    file_name: str,
    normalized_data_input_dir: str,
    output_dir: str,
    db_information: Dict,
    num_table_splits: int,
    column_stats: Dict[str, TableStats],
    scale_columns: Dict[str, set],
    new_varchar_lengths: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, int]]:
    file_path = os.path.join(normalized_data_input_dir, file_name)
    t = os.path.splitext(file_name)[0]

    if t not in db_information["table_col_info"]:
        log(f"Table {t} not found in schema, skipping.")
        return None

    # check if all scaled up versions of the table already exist - in this case we do not read the original table
    scale_candidates = []
    for i in range(num_table_splits):
        output_path = join(output_dir, f"{t}_{i}.csv")
        if not os.path.exists(output_path):
            scale_candidates.append(i)

    if len(scale_candidates) > 0:
        log(f"Reading original table {t} from {file_path}")
        orig_df_table = read_csv(
            db_information,
            file_path,
            t,
            use_custom_nan=True,
            use_dataset_specific_read_kwargs=False,
        )
    else:
        # data is already normalized
        orig_df_table = None

    # initialize new varchar lengths dictionary
    # this will be used to adjust the schema later on (due to scaling up varchar columns their lengths might change)
    new_varchar_lengths = dict()

    # generate additional rows by scaling up existing rows - will be set aside for inserts queries / ...
    if len(scale_candidates) > 0:
        log(f"Scaling table {t} (to num versions: {num_table_splits})")
        for i in tqdm(
            range(num_table_splits), desc=f"Scaling up table {t}", unit="version"
        ):
            output_path = join(output_dir, f"{t}_{i}.csv")
            if os.path.exists(output_path):
                log(f"Table {t}_{i} already exists, skipping.")
                continue
            log(f"Creating table {t}_{i} ({output_path})")

            curr_df_table = orig_df_table.copy(deep=True)

            for c in scale_columns[t]:
                dtype = retrieve_column_type(
                    db_information["table_col_info"][t][c]["type"]
                )

                if dtype in [ColumnType.INT, ColumnType.FLOAT]:
                    offset = find_numeric_offset(c, column_stats, db_information, t)
                    curr_df_table[c] += (i + 1) * offset

                elif dtype in [ColumnType.VARCHAR]:
                    curr_df_table[c] = curr_df_table[c].astype(str) + f"_{i}"

                    if i == num_table_splits - 1:
                        max_len = curr_df_table[c].astype(str).str.len().max()

                        if t not in new_varchar_lengths:
                            new_varchar_lengths[t] = dict()
                        new_varchar_lengths[t][c] = int(max_len)
                else:
                    log(curr_df_table[c].dtype)
                    raise NotImplementedError

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            null_representation = "<!NULL-?>"
            log(f"Writing out {t}_{i} to csv")
            curr_df_table.to_csv(
                output_path,
                mode="a",
                header=False,
                index=False,
                na_rep=null_representation,
                chunksize=100_000,  # Write in chunks if the DataFrame is large
            )
    else:
        log(f"All scaled up versions of table {t} already exist, skipping.")

    return new_varchar_lengths


def prepare_and_scale_dataset(
    normalized_data_input_dir: str,
    output_dir: str,
    new_schema_file_path: str,
    sql_schema_path: str,
    json_schema_path: str,
    statistics_path: str,
    schema_augmentation_factor: int,
):
    log(f"Preparing and scaling schema by {schema_augmentation_factor}x .")
    # get schema information
    sql_schema = get_sql_schema(sql_schema_path, keep_newline=True)
    db_information = get_json_schema(json_schema_path)
    column_stats = load_database_stats(statistics_path)

    # extract columns requiring scaling (pk / fk columns)
    scale_columns = extract_scale_columns(db_information)

    new_varchar_lengths_file_path = join(output_dir, "new_varchar_lengths.json")
    if os.path.exists(new_varchar_lengths_file_path):
        with open(new_varchar_lengths_file_path, "r") as f:
            new_varchar_lengths = json.load(f)
    else:
        new_varchar_lengths = dict()

    assert os.path.exists(normalized_data_input_dir), (
        f"Input directory {normalized_data_input_dir} does not exist."
    )

    scale_fn = functools.partial(
        scale_file,
        normalized_data_input_dir=normalized_data_input_dir,
        output_dir=output_dir,
        db_information=db_information,
        num_table_splits=schema_augmentation_factor,
        column_stats=column_stats,
        scale_columns=scale_columns,
        new_varchar_lengths=new_varchar_lengths,
    )

    # run this multiprocessing or not
    # if parallelize is True, we will use multiprocessing to scale the tables in parallel
    parallelize = True

    if parallelize:
        with multiprocessing.Pool() as pool:
            results = pool.map(scale_fn, os.listdir(normalized_data_input_dir))
    else:
        results = []
        for file_name in normalized_data_input_dir:
            result = scale_fn(file_name)
            results.append(result)

    # combine dicts
    for tmp in results:
        if not tmp:
            continue
        for t, cols in tmp.items():
            if t not in new_varchar_lengths:
                new_varchar_lengths[t] = dict()
            for c, length in cols.items():
                if c not in new_varchar_lengths[t]:
                    new_varchar_lengths[t][c] = length
                else:
                    new_varchar_lengths[t][c] = max(new_varchar_lengths[t][c], length)

    # save new varchar lengths
    with open(new_varchar_lengths_file_path, "w") as f:
        # we are caching this - sometimes we are skipping tables already scaled up - but still need this info to adjust the schema
        # so we save it here and can use it later
        json.dump(new_varchar_lengths, f, indent=4)

    # get dbms schema
    if "DROP TABLE IF EXISTS" in sql_schema:
        sql_schema_split = sql_schema.split("DROP TABLE IF EXISTS ")
    elif "drop table if exists" in sql_schema:
        sql_schema_split = sql_schema.split("drop table if exists ")
    else:
        raise NotImplementedError(
            "Expecting that DROP TABLE IF EXISTS is in the schema and can be used as separator"
        )

    # read schema
    new_schema = []
    for table_def in sql_schema_split:
        # remove leading and trailing whitespaces
        table_def = table_def.strip()

        # string will start with the remainder of 'DROP TABLE IF EXISTS' i.e. "table_name" ...
        if table_def.startswith('"'):
            # extract table name from remainder of "DROP TABLE IF EXISTS" line
            table_name = table_def.split("\n")[0].strip('"; ')

            # flatten the new_varchar_lengths dict to a dict with (table, column) as key
            new_varchar_lengths_flat = {
                (t, c): length
                for t, cols in new_varchar_lengths.items()
                for c, length in cols.items()
            }

            # find varchar columns of this table which need adjustment
            for (t, c), vc_length in new_varchar_lengths_flat.items():
                if t != table_name:
                    continue

                found = 0
                for search_string, repl_string in [
                    (rf'"{c}"\s+varchar\(\d+\)', rf'"{c}" varchar({vc_length:.0f})'),
                    (rf'"{c}"\s+char\(\d+\)', rf'"{c}" char({vc_length:.0f})'),
                    (
                        rf'"{c}"\s+character varying\(\d+\)',
                        rf'"{c}" character varying({vc_length:.0f})',
                    ),
                ]:
                    table_def = re.sub(search_string, repl_string, table_def)

                    new_found = table_def.count(repl_string)
                    assert new_found in [0, 1], (
                        f"Expected to find at most one column definition for {t}.{c} with matching type. Found: {new_found}. Search string: {search_string}, table_def: {table_def}"
                    )
                    found += new_found

                assert found == 1, (
                    f"Expected to find exactly one column definition for {t}.{c} with matching type. Found: {found}. table_def: {table_def}"
                )

        if table_def.strip() != "":
            table_name = table_def.split("\n")[0].strip(';')
            table_def = table_def + "\n\n"

            # Clean table name - remove quotes if present
            clean_table_name = table_name.strip('"')

            # add schema definitions for all multiversions of this table
            for i in range(schema_augmentation_factor):
                # Replace quoted table names
                table_def_split = table_def.replace(
                    f'"{clean_table_name}"', f'"{clean_table_name}_{i}"'
                )

                # Also handle unquoted table names in CREATE TABLE statements
                if f"CREATE TABLE {clean_table_name}\n" in table_def_split:
                    # we have an unescaped table name, we need to replace it manually
                    # replace the table name in the CREATE TABLE statement
                    table_def_split = table_def_split.replace(
                        f"CREATE TABLE {clean_table_name}", f'CREATE TABLE "{clean_table_name}_{i}"'
                    )

                new_schema.append(table_def_split)
        else:
            new_schema.append(table_def)

    pg_schema = "DROP TABLE IF EXISTS ".join(new_schema)

    with open(new_schema_file_path, "w") as file:
        file.write(pg_schema)


def get_dataset_size(data_dir: str, schema: Dict):
    filesize_b = 0
    for t in schema["tables"]:
        table_path = os.path.join(data_dir, f"{t}.csv")
        filesize_b += os.path.getsize(table_path)
    filesize_gb = filesize_b / (1024**3)
    return filesize_gb


def find_numeric_offset(
    c: str, column_stats: dict[str, TableStats], schema: Dict, t: str
):
    offset = None
    # if column is referencing, take this offset
    for t_out, col_out, t_in, col_in in schema["relationships"]:
        if t_out != t:
            continue

        if not isinstance(col_out, list):
            col_out = [col_out]
            col_in = [col_in]
        if c not in col_out:
            continue

        c_idx = col_out.index(c)

        offset = column_stats[t_in].columns[col_in[c_idx]].quantiles[100] + 1
    # take own offset if it is not referencing
    if offset is None:
        offset = column_stats[t].columns[c].quantiles[100] + 1
    return offset


def extract_scale_columns(schema: Dict):
    scale_columns = collections.defaultdict(set)
    for table_l, col_l, table_r, col_r in schema["relationships"]:
        if not isinstance(col_l, list):
            col_l = [col_l]
            col_r = [col_r]

        for table, columns in [(table_l, col_l), (table_r, col_r)]:
            for c in columns:
                scale_columns[table].add(c)

    # add pk columns to scale_columns
    for table, table_stats in schema["table_col_info"].items():
        for column, column_stats in table_stats.items():
            if column_stats["pk"]:
                scale_columns[table].add(column)
    return scale_columns
