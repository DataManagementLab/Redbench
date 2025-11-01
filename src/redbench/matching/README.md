# Matching Approach
This README provides basic instructions on how to use the matching based workload generator.

## Prepare Input Data
In the config you can specify the support benchmark to use. 
These queries will be used to match to.
We have provided you both TPCDS as well as JOB/STATS.
E.g. use one of the following config snippets for experimenting with them:
```json
"support_benchmarks": [
    {
        "id": "tpcds",
        "sf": 1,
        "name": "TPC-DS 1 GB",
        "stats_table": "tpcds_stats",
        "table_ids_table": "tpcds_table_ids",
        "override": false
    }
]
```

or for IMDB / JOB:

```json
"support_benchmarks": [
    {
        "id": "imdb",
        "name": "imdb",
        "stats_table": "imdb_stats",
        "table_ids_table": "imdb_table_ids",
        "override": false
    }
]
```
You can also add your own benchmark.

## Configuration
The repository uses a JSON configuration file to specify various parameters for data processing. Below is an explanation of each configuration option:

| Key | Description | Example  Value|
| :----- | :----- | :----- |
| `support_benchmarks` | The queries to map to. Can include multiple sources (must follow same db schema) | (see above) |
| `start_date`          | The starting date for a specific time range relevant to the data processing. The format should be `YYYY-MM-DD HH:MM:SS`.                                                            | `"2024-03-01 00:00:00"`                                 |
| `end_date`            | The ending date for the same time range. The format should be `YYYY-MM-DD HH:MM:SS`.                                                                                                |`"2024-05-01 00:00:00"`                                 |
| `limit_redset_rows_read` | Limit number of rows to be read from redset (for cluster/db) to speed up processing. This will act as natural limit for the amount of queries that can be produced. | `None` |
| `matching_method` | Which algorithm to use for matching. By default we do scanset-matching. The Redbench-v1 algorithm `join`-matching is available as well. Scanset does tries to match to the query with the most similar scanset while join-matching only looks for similar number of joins. | `scanset`|
| `redset_exclude_tables_never_read` | Exclude all (write-only) queries that write to a table that is never read - these queries might be just additional noise since they produce write-load without affecting caching/MVs/... To replicate the true redset this should be deactivated. | `false` |
| `use_table_versioning` | This option will rewrite queries to multiple universes to make sure that exactly one redset table will be mapped to each instance of the support benchmark. To facilitate this it will map several copies of the same table (`orders_0`, `orders_1`, ...). However this requires modifications to the schema, hence disabled by default.  | `false` |