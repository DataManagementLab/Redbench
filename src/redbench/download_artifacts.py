import argparse
import os
from typing import List

import osfclient
from tqdm import tqdm
from utils.log import log


def download_artifacts(
    artifacts_dir: str, databases: List[str], download_only_duckdb_file: bool = False
):
    # create the output directory if it does not exist
    output_path = artifacts_dir
    os.makedirs(output_path, exist_ok=True)

    # create connection to OSF
    osf = osfclient.OSF()
    project = osf.project("fgny3")
    log(f'Download files from OSF Project "{project.title}" (ID: {project.id}) - database, artifacts, ...')
    storage = project.storage()

    # download all files from osf
    files = list(storage.files)
    for file in tqdm(files, desc="Downloading files", total=len(files)):
        if not download_only_duckdb_file:
            log(f"Downloading file: {file.path} (size: {file.size / 1000000:.3f} MB)")
        file_path = file.path.strip(
            "/"
        )  # Remove leading slash for local path compatibility

        if file_path.startswith("example_databases") and file_path.endswith(".zip"):
            # make sure the file is in the support database list
            db_name = file_path.split("/")[1].replace(".zip", "")
            if db_name not in databases:
                # log(
                #     f"Skipping {file_path} as it is not in the support database list: {databases}"
                # )
                continue
        elif file_path.startswith("example_databases") and not file_path.endswith(
            ".zip"
        ):
            # extract the folder name
            folder_name = file_path.split("/")[1]
            if folder_name not in databases:
                # log(
                #     f"Skipping {file_path} as it is not in the support database list: {databases}"
                # )
                continue

        # apply renaming from example_databases to tmp_generation
        file_path = file_path.replace("example_databases", "tmp_generation")

        # construct the target file path
        target_file_path = os.path.join(output_path, file_path)

        # if only downloading duckdb files, skip other files
        if download_only_duckdb_file and not target_file_path.endswith(
            "db_original.duckdb"
        ):
            continue

        if os.path.exists(target_file_path):
            log(f"File {target_file_path} already exists, skipping download.")
            continue

        target_file_base_dir = os.path.dirname(target_file_path)
        os.makedirs(target_file_base_dir, exist_ok=True)

        with open(target_file_path, "wb") as f:
            file.write_to(f)

        # extract if it's a zip file
        if file_path.endswith(".zip"):
            import zipfile

            log("Unzipping...")
            with zipfile.ZipFile(target_file_path, "r") as zip_ref:
                zip_ref.extractall(target_file_base_dir)
            log(f"Extracted {file_path} to {target_file_base_dir}")

    log(f"All files downloaded to: {output_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download artifacts.")
#     parser.add_argument(
#         "--artifacts_dir", type=str, required=True, help="Path to the output directory."
#     )
#     parser.add_argument(
#         "--databases",
#         type=str,
#         nargs="+",
#         default=["baseball"],
#         help="List of support databases to use (default: baseball). Options: baseball, imdb",
#     )

#     args = parser.parse_args()

#     download_artifacts(args.artifacts_dir, args.databases)
