import ast
from typing import Dict, List
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table_experiments as dt
import io
import json
from lmfit.model import load_modelresult
from lmfit.model import save_modelresult
import numpy as np
import os
import pandas as pd
import plotly
import urllib
import urllib.parse

from diffcapanalyzer.databasewrappers import get_filename_pref
from diffcapanalyzer.databasewrappers import is_file_exists_in_db
from diffcapanalyzer.databasewrappers import process_data
from diffcapanalyzer.databasewrappers import macc_chardis
from diffcapanalyzer.databasefuncs import init_master_table
from diffcapanalyzer.databasefuncs import get_file_from_database
from diffcapanalyzer.chachifuncs import col_variables
from diffcapanalyzer.dataframe_tools import (
    expect_columns_of_dataframe,
    rename_columns_of_dataframe,
)
from diffcapanalyzer.descriptors import generate_model


def parse_contents(
    binary_data: bytes,
    filename: str,
    datatype: str,
    database,
    windowlength=9,
    polyorder=3,
):
    """Checks if the uploaded file exists in the database yet. Will
    process and add that file to the database if it doesn't appear in
    the master table yet. Otherwise will return html.Div that the
    file already exists in the database."""

    cleanset_name = get_filename_pref(filename) + "CleanSet"
    # this gets rid of any filepath in the filename and just leaves the
    # clean set name as it appears in the database check to see if the
    # database exists, and if it does, check if the file exists.
    file_exists_in_db = is_file_exists_in_db(database, filename)
    if file_exists_in_db:
        df_clean = get_file_from_database(cleanset_name, database)
        new_peak_thresh = 0.7
        feedback = generate_model(df_clean, filename, new_peak_thresh, database)
        return "That file exists in the database: " + str(get_filename_pref(filename))
    else:
        try:
            decoded_dataframe = decoded_to_dataframe(binary_data, datatype, filename)
            print("decoded")
            process_data(
                filename, database, decoded_dataframe, datatype, windowlength, polyorder
            )
            print("processed")
            df_clean = get_file_from_database(cleanset_name, database)
            print("cleaned")
            new_peak_thresh = 0.7
            feedback = generate_model(df_clean, filename, new_peak_thresh, database)
            print("generated")
            return "New file has been processed: " + str(get_filename_pref(filename))
        except Exception as e:
            return (
                "There was a problem uploading that file. "
                + "Check the format of the upload file is as expected."
                + str(e)
            )


def decoded_to_dataframe(binary_data: bytes, datatype: str, file_name: str):
    """Decodes the contents uploaded via the app. Returns
    contents as a dataframe. Accepts only csv files for
    both Arbin and MACCOR type data."""
    if binary_data is None:
        try:
            data1 = pd.read_csv(file_name, index_col=False)
        except BaseException:
            data1 = pd.read_csv(file_name, delimiter="\t", index_col=False)
    else:
        try:
            contents = io.StringIO(binary_data.decode("utf-8"))
        except BaseException:
            contents = io.StringIO(binary_data.decode("utf-16"))
        try:
            data1 = pd.read_csv(contents, index_col=False)
        except BaseException:
            data1 = pd.read_csv(contents, delimiter="\t", index_col=False)

    if datatype == "ARBIN":
        data1["datatype"] = "ARBIN"
        expected_cols = [
            "Cycle_Index",
            "Voltage(V)",
            "Current(A)",
            "Charge_Capacity(Ah)",
            "Discharge_Capacity(Ah)",
            "Step_Index",
        ]
        expect_columns_of_dataframe(expected_cols, data1)

    elif datatype == "MACCOR":
        data1["MaccCharLab"] = data1.apply(lambda row: macc_chardis(row), axis=1)
        data1["Current(A)"] = data1["Current [A]"] * data1["MaccCharLab"]
        data1["datatype"] = "MACCOR"

        expected_cols = ["Cycle C", "Voltage [V]", "Current [A]", "Cap. [Ah]", "Md"]
        expect_columns_of_dataframe(expected_cols, data1)
        columns_map = {
            "Cycle C": "Cycle_Index",
            "Voltage [V]": "Voltage(V)",
            "Current [A]": "Abs_Current(A)",
            "Cap. [Ah]": "Cap(Ah)",
        }
        rename_columns_of_dataframe(columns_map, data1)

    elif datatype == "FE-Neware":
        data1["datatype"] = "FE-Neware"

        # TODO (cshen): We should scale the input correctly here.
        # By this time, raw capacity number and energy number are not scaled correctly within our processed data.

        print("Formatted as FE-Neware")
        print([item.title() for item in data1.columns])

    else:
        None
    return data1


def pop_with_db(filename: str, database):
    """Returns dataframes that can be used to populate the app graphs.
    Finds the already existing file in the database and returns
    the cleaned version (as a dataframe) and the raw version
    (also as a dataframe)."""
    cleanset_name = get_filename_pref(filename) + "CleanSet"
    rawset_name = get_filename_pref(filename) + "Raw"
    if is_file_exists_in_db(database, filename):
        # then the file exists in the database and we can just read it
        df_clean = get_file_from_database(cleanset_name, database)
        df_raw = get_file_from_database(rawset_name, database)
        datatype = df_clean["datatype"].iloc[0]
        (
            cycle_ind_col,
            data_point_col,
            volt_col,
            curr_col,
            dis_cap_col,
            char_cap_col,
            charge_or_discharge,
        ) = col_variables(datatype)

    else:
        df_clean = None
        df_raw = None
        peakloc_dict = {}

    return df_clean, df_raw
