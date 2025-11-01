import pandas as pd


def build_insert_query():
    pass


def read_csv_subset(file_path, start_index, length):
    """
    Reads a subset of rows from a CSV file efficiently.

    Parameters:
        file_path (str): Path to the CSV file.
        start_index (int): The starting index (0-based) of the subset.
        length (int): The number of rows to read.

    Returns:
        pd.DataFrame: A DataFrame containing the requested subset of rows.
    """
    # Use skiprows to skip rows before the start index, and nrows to read only the required number of rows

    # Handle NULL values in the DataFrame
    df = pd.read_csv(file_path, skiprows=range(1, start_index + 1), nrows=length)
    df = df.replace("<!NULL-?>", None)
    return df


def generate_insert_query(table_name: str, df: pd.DataFrame) -> str | None:
    """
    Generate an SQL INSERT query for a given DataFrame.

    Args:
        table_name (str): The name of the SQL table.
        df (pd.DataFrame): The DataFrame to generate the query for.

    Returns:
        str: The generated SQL INSERT query.
    """

    # Extract column names and prepare the column part of the query
    columns = ", ".join([f'"{col}"' for col in df.columns])

    # Prepare the values part of the query
    values = []
    for _, row in df.iterrows():
        row_values = []
        for value in row:
            if (
                value is None or value == "nan" or pd.isna(value)
            ):  # Replace Python None with SQL NULL
                row_values.append("NULL")
            elif isinstance(value, str):  # Quote string values
                value_with_replacement = value.replace("'", "''")
                row_values.append(f"'{value_with_replacement}'")
            else:  # Handle numeric values directly
                row_values.append(str(value))
        values.append(f"({', '.join(row_values)})")
    if not values:
        return None
    values_part = ", ".join(values)
    # Combine everything into a single SQL query
    query = f'INSERT INTO "{table_name}" ({columns}) VALUES {values_part};'

    return query
