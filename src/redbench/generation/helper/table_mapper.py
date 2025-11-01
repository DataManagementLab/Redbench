from typing import Dict

from redbench.generation.dataset_input.retrieve_statistics import TableStats


def map_redset_table_to_physical_table(redset_table_id, physical_tables):
    num_tables = len(physical_tables)

    return physical_tables[redset_table_id % num_tables]


def map_redset_table_to_physical_table_by_table_sizes(
    redset_tables: Dict[int, float], physical_tables: Dict[str, TableStats]
) -> Dict[int, str]:
    # sort physical tables by size descending
    physical_table_names = list(physical_tables.keys())
    physical_table_names.sort(
        key=lambda table: physical_tables[table].total_rows, reverse=True
    )

    # sort redset tables by size descending
    redset_table_names = list(redset_tables.keys())
    redset_table_names.sort(key=lambda table: redset_tables[table], reverse=True)

    # compute num physical tables per redset table
    num_phys_table_per_redset_table = max(1, len(redset_tables) / len(physical_tables))

    # map largest redset tables to largest physical tables
    mapping = {}
    for idx, key_a in enumerate(redset_table_names):
        src_idx = int(idx / num_phys_table_per_redset_table)
        assert src_idx < len(physical_table_names), (
            f"Index {src_idx} out of bounds for physical tables of size {len(physical_table_names)} / {len(redset_tables)} / {num_phys_table_per_redset_table}"
        )
        mapping[key_a] = physical_table_names[src_idx]
    return mapping


# if __name__ == "__main__":
#     query_data = {
#         "query_type": "SELECT",
#         "num_joins": 2,
#         "max_mbytes_scanned": 500
#     }
#     with open("<path>/statistics_output.json", "r", encoding="utf-8") as file:
#         data = json.load(file)

#     redset_tables = {
#         9679: 753541.0, 9675: 393684.0, 8957: 376798.0, 1752: 375831.0, 1606: 369657.0, 39: 226713.0, 56: 223376.0,
#         1143: 126460.0, 9308: 126105.0, 9552: 125753.0, 4323: 122723.0, 4629: 109938.0, 9549: 107622.0, 4622: 76718.0,
#         8963: 76530.0, 4321: 74854.0, 9558: 71119.0, 8958: 70835.0, 115: 48512.0, 124: 42752.0, 1139: 36974.0,
#         36: 32276.0, 45: 29696.0, 69: 29696.0, 106: 28672.0, 9305: 26712.0, 9553: 26634.0, 8954: 26583.0,
#         1761: 26556.0, 1757: 26531.0, 9672: 26148.0, 37: 25801.0, 100: 23936.0, 46: 20864.0, 1763: 19094.0,
#         77: 18618.0, 9560: 17717.0, 8961: 17673.0, 16: 15872.0, 62: 15866.0, 1: 12639.0, 13: 12346.0,
#         1770: 12220.0, 64: 11648.0, 43: 11320.0, 23: 11020.0, 113: 11008.0, 8: 10703.0, 104: 10116.0, 111: 8871.0,
#         5: 8240.0, 34: 7835.0, 112: 7297.0, 12: 7066.0, 9691: 6878.0, 94: 6784.0, 108: 6464.0, 10: 6386.0,
#         83: 6239.0, 9307: 6093.0, 4624: 6074.0, 20: 6065.0, 41: 5988.0, 258: 5974.0, 42: 5888.0, 1600: 5306.0,
#         9681: 4428.0, 31: 4185.0, 87: 3972.0, 7: 3672.0, 121: 3451.0, 17: 3368.0, 68: 2615.0, 1611: 2496.0,
#         9310: 2444.0, 118: 2322.0, 28: 1921.0, 65: 1884.0, 22: 1882.0, 18: 1782.0, 117: 1656.0, 66: 1182.0,
#         1612: 1142.0, 61: 1139.0, 264: 1134.0, 9564: 1114.0, 265: 1099.0, 29: 416.0, 33: 376.0, 48: 371.0, 19: 210.0,
#         26: 77.0, 30: 62.0, 109: 30.0, 47: 0.0, 63: 0.0, 85: 0.0
#     }


#     id = 2349
#     log(map_redset_table_to_physical_table(id, list(data.keys())))
#     log(map_redset_table_to_physical_table_sorted(redset_tables, data))
