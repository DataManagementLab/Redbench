import multiprocessing
import os

from tqdm import tqdm

from redbench.generation.dataset_input.load_schema import get_json_schema
from redbench.generation.dataset_input.read_csv import read_csv
from redbench.utils.log import log


def process_table(args):
    t, input_dir, output_path, db_information, force = args
    full_out_path = os.path.join(output_path, f"{t}.csv")
    if not force and os.path.exists(full_out_path):
        log(f"Table {t} already exists, skipping.")
        return
    log(f"Creating normalized table {t} for {full_out_path}")
    file_path = os.path.join(input_dir, f"{t}.csv")
    os.makedirs(os.path.join(output_path), exist_ok=True)
    orig_df_table = read_csv(
        db_information,
        file_path,
        t,
        use_custom_nan=False,
        use_dataset_specific_read_kwargs=True,
    )
    orig_df_table.to_csv(
        full_out_path, mode="w", header=True, index=False, na_rep="<!NULL-?>"
    )


def comp(input_dir: str, output_path: str, json_schema_path: str, force=False):
    db_information = get_json_schema(json_schema_path)
    tables = list(db_information["table_col_info"].keys())

    args_list = [(t, input_dir, output_path, db_information, force) for t in tables]

    with multiprocessing.Pool() as pool:
        list(
            tqdm(
                pool.imap(process_table, args_list),
                total=len(tables),
                desc="Processing tables",
            )
        )


def create_normalized_dataset(
    input_dir, output_dir, json_schema_path: str, force=False
):
    log("Creating normalized dataset...")
    comp(
        input_dir=input_dir,
        output_path=output_dir,
        json_schema_path=json_schema_path,
        force=force,
    )
    log("Finished creating normalized dataset.")
