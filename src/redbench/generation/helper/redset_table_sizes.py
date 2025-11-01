from collections import defaultdict

import pandas as pd


def define_sizes_for_redset_tables(workload: pd.DataFrame):
    """
    Extract rough relative size estimates between tables in the workload.
    This assumes that the bytes read from a table are a rough proxy for its size relative to other tables - realistic absolute sizes is not important here.
    The size is determined by the maximum MB scanned for each table.
    This is derived from single table queries - here we can exactly attribute the scan size to the table.
    """
    table_sizes = defaultdict(float)
    all_table_ids = set()
    write_all_table_ids = set()

    for _, row in workload.iterrows():
        read_table_ids = (
            str(row["read_table_ids"]) if pd.notna(row["read_table_ids"]) else ""
        )
        max_mbytes_scanned = row["max_mbytes_scanned"]

        # parse table ids (comma separated)
        table_ids = read_table_ids.split(",")

        # collect all table ids
        all_table_ids.update(table_ids)

        write_table_ids = (
            str(row["write_table_ids"]) if pd.notna(row["write_table_ids"]) else ""
        )
        write_table_ids = write_table_ids.split(",")
        write_all_table_ids.update(write_table_ids)

        # If there is only one table ID and it's a number
        if len(table_ids) == 1 and table_ids[0].isdigit():
            table_id = int(table_ids[0])
            table_sizes[table_id] = max(table_sizes[table_id], max_mbytes_scanned)

    table_sizes = dict(
        sorted(table_sizes.items(), key=lambda item: item[1], reverse=True)
    )

    # convert keys to int
    all_table_ids = {tid.strip() for tid in all_table_ids if tid.strip() != ""}
    write_all_table_ids = {
        tid.strip() for tid in write_all_table_ids if tid.strip() != ""
    }

    all_table_ids = {int(tid) for tid in all_table_ids}
    write_all_table_ids = {int(tid) for tid in write_all_table_ids}

    return table_sizes, all_table_ids, write_all_table_ids
