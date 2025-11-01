import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ColumnInfo:
    type: str
    pk: bool


# Define the structure for relationships
@dataclass
class Relationship:
    table_1: str
    table_1_columns: List[str]
    table_2: str
    table_2_columns: List[str]


# Define the structure for the "accidents" schema
@dataclass
class Schema:
    name: str
    relationships: List[Relationship]
    table_col_info: Dict[str, Dict[str, ColumnInfo]]


def load_schema_from_file(file_path: str):
    # Open and read the JSON file
    with open(file_path, "r") as f:
        data = json.load(f)

    # Check for required keys in the top-level schema
    required_keys = ["name", "relationships", "table_col_info"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Missing required key: {key}")

    # Validate the type of 'name' (should be a string)
    if not isinstance(data["name"], str):
        raise TypeError(f"Expected 'name' to be a string, but got {type(data['name'])}")

    # Validate relationships
    if not isinstance(data["relationships"], list):
        raise TypeError(
            f"Expected 'relationships' to be a list, but got {type(data['relationships'])}"
        )

    for rel in data["relationships"]:
        if not isinstance(rel, list) or len(rel) != 4:
            raise ValueError(
                f"Each relationship should be a list with 4 elements, but got {rel}"
            )

        # Validate first and third elements are strings
        if not isinstance(rel[0], str):
            raise TypeError(
                f"Expected first element of relationship to be a string, but got {type(rel[0])}"
            )
        if not isinstance(rel[2], str):
            raise TypeError(
                f"Expected second element of relationship to be a string, but got {type(rel[2])}"
            )

        # Validate second and fourth elements are lists of strings
        if not isinstance(rel[1], list) or not all(
            isinstance(val, str) for val in rel[1]
        ):
            raise TypeError(
                f"Expected third element of relationship to be a list of strings, but got {type(rel[1])}"
            )
        if not isinstance(rel[3], list) or not all(
            isinstance(val, str) for val in rel[3]
        ):
            raise TypeError(
                f"Expected fourth element of relationship to be a list of strings, but got {type(rel[3])}"
            )

    # # Convert the relationships into Relationship objects
    # relationships = [
    #     Relationship(
    #         table_1=rel[0],
    #         table_1_columns=rel[1],
    #         table_2=rel[2],
    #         table_2_columns=rel[3],
    #     )
    #     for rel in data["relationships"]
    # ]

    # Validate table_col_info structure
    if not isinstance(data["table_col_info"], dict):
        raise TypeError(
            f"Expected 'table_col_info' to be a dictionary, but got {type(data['table_col_info'])}"
        )

    table_col_info = {}
    for table_name, columns in data["table_col_info"].items():
        if not isinstance(columns, dict):
            raise TypeError(
                f"Expected 'columns' to be a dictionary for table '{table_name}', but got {type(columns)}"
            )

        table_col_info[table_name] = {}
        for column_name, column in columns.items():
            if not isinstance(column, dict):
                raise TypeError(
                    f"Expected 'column' to be a dictionary for column '{column_name}' in table '{table_name}', but got {type(column)}"
                )

            # Validate column info structure
            if "type" not in column or "pk" not in column:
                raise KeyError(
                    f"Missing 'type' or 'pk' in column info for '{column_name}' in table '{table_name}'"
                )

            if not isinstance(column["type"], str):
                raise TypeError(
                    f"Expected 'type' to be a string, but got {type(column['type'])}"
                )
            if not isinstance(column["pk"], bool):
                raise TypeError(
                    f"Expected 'pk' to be a bool, but got {type(column['pk'])}"
                )

            table_col_info[table_name][column_name] = ColumnInfo(
                type=column["type"], pk=column["pk"]
            )

    # Return the Schema object
    return data
