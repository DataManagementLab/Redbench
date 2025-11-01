from enum import Enum


class ColumnType(Enum):
    VARCHAR = "VARCHAR"
    INT = "INT"
    FLOAT = "FLOAT"
    DATE = "DATE"
    UNKNOWN = "UNKNOWN"


def retrieve_column_type(column_type:str) -> ColumnType:
    data_type = column_type.upper()
    if any(numeric_type in data_type for numeric_type in ("INTEGER", "BIGINT", "INT")):
        return ColumnType.INT
    if any(numeric_type in data_type for numeric_type in ("DOUBLE", "REAL", "DECIMAL", "NUMERIC", "FLOAT")):
        return ColumnType.FLOAT
    if any(numeric_type in data_type for numeric_type in ("VARCHAR", "TEXT", "STRING", "CHAR")):
        return ColumnType.VARCHAR
    if any(numeric_type in data_type for numeric_type in ("DATE", "TIMESTAMP", "TIME")):
        return ColumnType.VARCHAR

    return ColumnType.UNKNOWN
