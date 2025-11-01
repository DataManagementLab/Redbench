from enum import Enum


class DataType(Enum):
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    STRING = "string"
    DATE = "date"
    TIME = "time"
    MISC = "misc"

    def __str__(self):
        return self.value

    def from_str(s: str):
        if s in ["int", "integer", "BIGINT"]:
            return DataType.INT
        if s in ["float", "double"] or s.startswith("decimal("):
            return DataType.FLOAT
        if s in ["categorical"]:
            return DataType.CATEGORICAL
        if (
            s in ["string", "varchar", "text"]
            or s.startswith("varchar")
            or s.startswith("char")
        ):
            return DataType.STRING
        if s in ["misc", "bytea"]:
            return DataType.MISC
        if s in ["date", "datetime"]:
            return DataType.DATE
        if s in ["time"]:
            return DataType.TIME
        raise ValueError(f"Unknown data type {s}")

    def get_pandas_dtype(self):
        if self == DataType.INT:
            return "Int64"
        if self == DataType.FLOAT:
            return "float"
        if self == DataType.CATEGORICAL:
            raise ValueError("Categorical data type has no pandas dtype")
        if self == DataType.STRING:
            return "string"
        if self == DataType.DATE:
            return "string"
        if self == DataType.TIME:
            return "string"
        if self == DataType.MISC:
            return "object"
        raise ValueError(f"Unknown data type {self}")
