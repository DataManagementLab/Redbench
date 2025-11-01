import os
from typing import Dict

import pandas as pd

from redbench.generation.dataset_input.DataType import DataType
from redbench.utils.log import log


def read_csv(
    schema: Dict,
    data_dir: str,
    table_name: str,
    use_custom_nan: bool = True,
    use_dataset_specific_read_kwargs: bool = False,
) -> pd.DataFrame:
    table_dir = data_dir
    assert os.path.exists(data_dir), f"Could not find table csv {table_dir}"

    # extract and parse datatype
    pd_dtype_dict = dict()
    for c, col_info in schema["table_col_info"][table_name].items():
        data_type = col_info["type"]
        data_type = DataType.from_str(data_type)
        pd_dtype_dict[c] = data_type.get_pandas_dtype()

    # read csv, enforce data types (pandas sometimes guesses wrong)
    try:
        if use_custom_nan:
            # use the special NaN keyword we introduced with the scaling code
            custom_nan_values = ["<!NULL-?>"]
        else:
            # custom nan values since NA appears in some datasets and is not nan
            custom_nan_values = [
                "",
                "#N/A",
                "#N/A N/A",
                "#NA",
                "-1.#IND",
                "-1.#QNAN",
                "-NaN",
                "-nan",
                "1.#IND",
                "1.#QNAN",
                "<NA>",
                "N/A",
                "NULL",
                "NaN",
                "n/a",
                "nan",
                "null",
            ]

            if schema["name"] == "imdb":
                custom_nan_values.remove(
                    "N/A"
                )  # N/A is a valid value in imdb e.g. it appears in the non-null column: title.title
            elif schema["name"] == "accidents":
                custom_nan_values.remove("")

        if use_dataset_specific_read_kwargs:
            args = schema["csv_kwargs"]
        else:
            # csv schema has been normalized, no need to use dataset specific read kwargs
            args = dict()

        df_table = pd.read_csv(
            table_dir,
            dtype=pd_dtype_dict,
            keep_default_na=False,
            na_values=custom_nan_values,
            **args,
        )

    except Exception as e:
        log(
            f"Could not read csv {table_dir} with csv_kargs: {schema['csv_kwargs']}, dtype: {pd_dtype_dict}"
        )
        raise e

    return df_table
