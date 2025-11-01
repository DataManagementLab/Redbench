# Generation Approach
This README provides basic instructions on how to use the generation based workload generator.

## Prepare Input Data:
* **Raw Database Tables (CSV):** Ensure you have a directory containing your database tables stored as individual `.csv` files. Each `.csv` file should represent a single table.
* **Redset Dataset File (Parquet):** You need to have a pre-existing dataset file in the `.parquet` format.

## Configuration
The repository uses a JSON configuration file to specify various parameters for data processing. Below is an explanation of each configuration option:

| Key                   | Description                                                                                                                                                                         | Example Value                                            |
| :-------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| :------------------------------------------------------- |
| `raw_database_tables` | The path to the directory containing your raw database tables in `.csv` format.                                                                                                     | `"/path/to/raw_datasets/baseball"` |
| `database_name`       | A name to identify your database. This can be used for organizational purposes in the output.                                                                                       | `"baseball"`                                           |
| `schema_augmentation_factor`           | The number of partitions. More partitions lead to more tables i.e. we can better match the redset to this schema. (1x means no additional tables generated)                                                                  | `2`                                                    |
| `repetition_exponent` | Exponent on how likely it is to sample small/large QIGs. Default is 1.                                                                                                              | `1`                                                    |
| `seed`                | A random seed value used for reproducibility of any random operations within the repository. Using the same seed will produce the same results for operations involving randomness. | `42`                                                 |
| `enable_random_table_ids` | A boolean flag indicating whether to assign random IDs to tables. `true` enables random IDs, `false` uses a redset tableset.                                                        | `false`                                                |
| `enable_random_databases` | A boolean flag indicating whether to introduce randomness related to databases from redset                                                                                          | `false`                                                |
| `add_conflict_logic`  | A boolean flag indicating whether to include logic for handling potential conflicts during data processing.                                                                         | `true`                                                 |
| `apply_sampling`      | Apply sampling on the query-instance-groups (QIGs) of the redset. If `false` all redset entries will be mapped. | `true`|
| `sample_size`         | The number of QIGs to sample from the redset.                                                                                                                              | `10000`                                                 |
| `max_size_qig`        | The maximum size allowed for QIGs. That can be helpfull to filter out queries with 1 million repetitions ;)                                                                         | `20000`                                                |
| `deactivate_repeating_inserts` | A boolean flag to deactivate the repetition of insert quries.                                                                                                                       | `false`                                                |
| `force_setup_creation` | A boolean flag to force the creation of a setup, even if it already exists.                                                                                                         | `false`                                                |
| `force_workload_creation` | A boolean flag to force the creation of a workload, even if it already exists                                             | `false`                                                |
| `start_date`          | The starting date for a specific time range relevant to the data processing. The format should be `YYYY-MM-DD HH:MM:SS`.                                                            | `"2024-03-01 00:00:00"`                                 |
| `end_date`            | The ending date for the same time range. The format should be `YYYY-MM-DD HH:MM:SS`.                                                                                                | `"2024-05-01 00:00:00"`                                 |
| `limit_redset_rows_read` | Limit number of rows to be read from redset (for cluster/db) to speed up processing. This will act as natural limit for the amount of queries that can be produced. | `None` |
| `include_copy` | Generate copy queries as well. For this we will import one of the scaled up versions of the base tables. We cannot ensure Primary-Key constraints here since the same csv might be loaded multiple times! Only recommended to use only a single copy occurs in the workload. | `false` |
| `include_analyze` | Generate queries that can represent the load of an analyze. These analyze queries stem from maintenance in redshift (updating statistics) and hence are usually not of interest since they are DBMS specific.  | `false` |
| `interpret_deviating_mbytes_as_structural_repetition`| Interpret large deviations in mbytes scanned of a repeating query, only as structural repetition (i.e. query template with different filter literals). If `false` structural repetition will not be enforced, however will still occur due to randomness in query generation. Default is `true`. | `true` |
| `redset_exclude_tables_never_read` | Exclude all (write-only) queries that write to a table that is never read - these queries might be just additional noise since they produce write-load without affecting caching/MVs/... To replicate the true redset this should be deactivated. | `false` |
| `validate_query_produces_rows` | Validate that generated queries always produce at least one row. This happens during query generation. This makes query generation significantly slower since queries will be executed multiple times in case of retries. | `true` |
| `min_table_scan_selectivity` | Ensures a minimum selectivity of table scans. The selectivity are approximated from redset bytes-read information but can lead to super small selecitivities - hence this option to set a larger minimum | `0.1` |
| `start_mapping_largest_table` | Start with the random walks from the approximately largest table of the read-set defined by the redset. If False: start with a random table from the read-set | `true` |

**To use this repository:**

1.  Create a JSON file (e.g., `config.json`) and fill it with the appropriate values based on your data and desired settings.
2.  Refer to the repository's specific scripts or modules to understand how this configuration file is loaded and used. You will likely need to modify or execute Python scripts within the repository, providing the path to your configuration file as an argument if necessary.

**Example `config.json`:**

```json
{
    "raw_database_tables": "/path/to/csv/baseball",
    "database_name": "baseball",
    "num_split": 2,
    "repetition_exponent": 1,
    "seed": 42,
    "enable_random_table_ids": false,
    "enable_random_databases": false,
    "add_conflict_logic": true,
    "apply_sampling": false,
    "max_size_qig": 100000,
    "deactivate_repeating_inserts": false,
    "force_setup_creation": false,
    "force_workload_creation": false,
    "start_date": "2024-03-01 00:00:00",
    "end_date": "2024-05-01 00:00:00"
}
```