from typing import Dict, List

import pandas as pd


def expect_columns_of_dataframe(expected_cols: List[str], dataframe: pd.DataFrame):
    assert all(item in list(dataframe.columns) for item in expected_cols)


def rename_columns_of_dataframe(columns_map: Dict[str, str], dataframe: pd.DataFrame):
    dataframe.rename(
        columns=columns_map,
        inplace=True,
    )


def rename_columns_of_dataframe(
    columns: List[str], rename_to: List[str], dataframe: pd.DataFrame
):
    columns_map = {item[0]: item[1] for item in zip(columns, rename_to)}
    rename_columns_of_dataframe(columns_map, dataframe)
