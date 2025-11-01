import os
import gzip
import orjson


def get_json_schema(path: str):
    assert os.path.exists(path), f"Could not find schema.json ({path})"

    # read file
    schema = load_json(path)
    return schema


def get_sql_schema(path: str, keep_newline: bool = False):
    with open(path, "r") as file:
        data = file.read()
        if not keep_newline:
            data = data.replace("\n", "")

    return data


def load_json(path: str, namespace: bool = False):
    """
    Read json from file
    :param path: path to json file
    :param namespace: return as namespace
    """
    if not os.path.isfile(path):
        raise Exception(f"Loading file failed - file {path} not found")

    assert not namespace, (
        f"Namespace {namespace} not supported - this is not JoRo approved!!"
    )

    if path.endswith(".json"):
        with open(path, "rb") as json_file:
            json_obj = orjson.loads(json_file.read())
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt") as json_file:
            json_obj = orjson.loads(json_file.read())
    else:
        raise Exception(f"Expected json file in file_path {path}")

    return json_obj
