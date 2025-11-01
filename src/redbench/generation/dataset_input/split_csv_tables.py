import csv
import os
import random
from os.path import isfile, join

from redbench.utils.log import log


def split_csv(file_path, output_dir, n_splits=2, distribution=None):
    """
    Splits a single CSV file into n sets with random row assignments.

    :param file_path: The path to the CSV file to split.
    :param output_dir: The directory where the split files will be saved.
    :param n_splits: The number of splits (default is 2).
    :param distribution: A list of floats defining the distribution of data across splits (optional).
                         If None, data will be distributed equally.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, mode="r") as file:
        reader = csv.reader(file)
        headers = next(reader)
        rows = list(reader)

    total_rows = len(rows)

    # If no distribution is provided, distribute equally
    if distribution is None:
        distribution = [1 / n_splits] * n_splits

    # Ensure the distribution sums to 1
    if abs(sum(distribution) - 1) > 1e-6:
        raise ValueError("Distribution must sum to 1.")

    # Shuffle rows to ensure random assignment
    random.shuffle(rows)

    # Calculate split points based on the distribution
    split_points = []
    cumulative = 0
    for frac in distribution:
        cumulative += frac
        split_points.append(int(cumulative * total_rows))

    # Split the data into sets based on the calculated split points
    split_data = []
    start_idx = 0
    for end_idx in split_points:
        split_data.append(rows[start_idx:end_idx])
        start_idx = end_idx

    # Write the splits to separate CSV files
    base_name = os.path.splitext(os.path.basename(file_path))[
        0
    ]  # Get filename without extension
    file_output_dir = join(output_dir, base_name)  # Create subdirectory for this file
    os.makedirs(file_output_dir, exist_ok=True)

    for split_id, data in enumerate(split_data):
        output_path = join(file_output_dir, f"{base_name}_{split_id + 1}.csv")
        with open(output_path, mode="w", newline="") as split_file:
            writer = csv.writer(split_file)
            writer.writerow(headers)  # Write headers
            writer.writerows(data)  # Write data for this split
        log(f"Created split {split_id + 1} at {output_path}")


def split_all_csvs(input_dir, output_dir, n_splits=2, distribution=None):
    """
    Applies split_csv to all CSV files in a directory.

    :param input_dir: Directory containing CSV files to split.
    :param output_dir: Directory where split files will be saved.
    :param n_splits: Number of splits per file (default is 2).
    :param distribution: List of floats defining the distribution of data across splits.
                         If None, data will be distributed equally.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        file_path = join(input_dir, file_name)
        if isfile(file_path) and file_name.endswith(".csv"):
            log(f"Processing {file_name}...")
            split_csv(file_path, output_dir, n_splits, distribution)
