import glob
from math import isclose
import numpy as np
import os
import pandas as pd
import requests
import scipy.io
import scipy.signal

from diffcapanalyzer.databasefuncs import get_file_from_database
from diffcapanalyzer.databasefuncs import update_database_newtable


def load_sep_cycles(core_file_name, database_name, datatype):
    """Loads cycles from an existing uploaded file from the
    database, and saves them as separate dataframes
    with the cycle number as the key."""
    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)
    name = core_file_name + "Raw"
    df_single = get_file_from_database(name, database_name)
    gb = df_single.groupby(by=[cycle_ind_col])
    cycle_dict = dict(iter(gb))
    for key in cycle_dict.keys():
        cycle_dict[key]["Battery_Label"] = core_file_name
        update_database_newtable(
            cycle_dict[key], core_file_name + "-" + "Cycle" + str(key), database_name
        )
    return cycle_dict


def get_clean_cycles(
    cycle_dict, core_file_name, database_name, datatype, windowlength=9, polyorder=3
):
    """Imports all separated out cycles in given path and cleans them
    and saves them in the database"""

    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)

    clean_cycle_dict = {}
    for i in sorted(list(cycle_dict.keys())):
        print(i, len(cycle_dict))
        print(i, cycle_dict.keys())
        charge, discharge = clean_calc_sep_smooth(
            cycle_dict[i], datatype, windowlength, polyorder
        )
        print(i, "calc")
        clean_data = charge.append(discharge, ignore_index=True)
        print(i, "clean")
        clean_data = clean_data.sort_values([data_point_col], ascending=True)
        print(i, "sort")
        clean_data = clean_data.reset_index(drop=True)
        print(i, "reset")
        cyclename = core_file_name + "-CleanCycle" + str(i)
        print(i, "name")
        clean_cycle_dict.update({cyclename: clean_data})
        print(i, "dict update")
        update_database_newtable(clean_data, cyclename, database_name)
        print(i, "update datatable")
        print(i, len(cycle_dict))
    return clean_cycle_dict


def get_clean_sets(clean_cycle_dict, core_file_name, database_name):
    """Imports all clean cycles of data from import path and appends
    them into complete sets of battery data, saved into save_filepath"""

    clean_set_df = pd.DataFrame()
    for k, v in clean_cycle_dict.items():
        clean_set_df = clean_set_df.append(v, ignore_index=True)
    update_database_newtable(clean_set_df, core_file_name + "CleanSet", database_name)
    return clean_set_df


############################
# Component Functions
############################


def clean_calc_sep_smooth(dataframe, datatype, windowlength, polyorder):
    """Takes one cycle dataframe, calculates dq/dv, cleans the data,
    separates out charge and discharge, and applies sav-golay filter.
    Returns two dataframes, one charge and one discharge.
    Windowlength and polyorder are for the sav-golay filter."""
    print("clean_calc_sep_smooth", type(dataframe))
    assert isinstance(dataframe, pd.DataFrame)
    print("clean_calc_sep_smooth", "assert")
    df = init_columns(dataframe, datatype)
    print("clean_calc_sep_smooth", "init columns")
    df1 = calc_dq_dqdv(df, datatype)
    print("clean_calc_sep_smooth", "calc dqdv")
    cleandf2 = drop_inf_nan_dqdv(df1, datatype)
    print("clean_calc_sep_smooth", "drop inf nan")
    charge, discharge = sep_char_dis(cleandf2, datatype)
    print("clean_calc_sep_smooth", "set_char_dis")
    # separating into charge and discharge cycles
    charge = clean_charge_discharge_separately(charge, datatype)
    print("clean_calc_sep_smooth", "clean charge")
    discharge = clean_charge_discharge_separately(discharge, datatype)
    print("clean_calc_sep_smooth", "clean discharge")

    if len(discharge) > windowlength:
        smooth_discharge = my_savgolay(discharge, windowlength, polyorder)
        print("clean_calc_sep_smooth", "discharge my_savgolay")
    else:
        discharge["Smoothed_dQ/dV"] = discharge["dQ/dV"]
        smooth_discharge = discharge
        print("clean_calc_sep_smooth", "discharge other")
    # this if statement is for when the datasets have less datapoints
    # than the windowlength given to the sav_golay filter.
    # without this if statement, the sav_golay filter throws an error
    # when given a dataset with too few points. This way, we simply
    # forego the smoothing function.
    if len(charge) > windowlength:
        smooth_charge = my_savgolay(charge, windowlength, polyorder)
        print("clean_calc_sep_smooth", "charge my_savgolay")
    else:
        charge["Smoothed_dQ/dV"] = charge["dQ/dV"]
        smooth_charge = charge
        print("clean_calc_sep_smooth", "charge other")
    # same as above, but for charging cycles.
    return smooth_charge, smooth_discharge


def init_columns(cycle_df, datatype):
    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)
    assert isinstance(cycle_df, pd.DataFrame)
    assert volt_col in cycle_df.columns
    assert dis_cap_col in cycle_df.columns
    assert char_cap_col in cycle_df.columns

    cycle_df = cycle_df.reset_index(drop=True)
    cycle_df["dV"] = None
    cycle_df["Discharge_dQ"] = None
    cycle_df["Charge_dQ"] = None
    return cycle_df


def calc_dq_dqdv(cycle_df, datatype):
    try:

        (
            cycle_ind_col,
            data_point_col,
            volt_col,
            curr_col,
            dis_cap_col,
            char_cap_col,
            charge_or_discharge,
        ) = col_variables(datatype)

        pd.options.mode.chained_assignment = None
        # to avoid the warning

        cycle_df["roundedV"] = round(cycle_df[volt_col], 3)
        cycle_df = cycle_df.drop_duplicates(
            subset=["roundedV", cycle_ind_col, charge_or_discharge]
        )
        cycle_df = cycle_df.reset_index(drop=True)
        cycle_df["dV"] = cycle_df[volt_col].diff()

        cycle_df_charge = cycle_df[cycle_df[curr_col] > 0]
        cycle_df_charge["Charge_dQ"] = cycle_df_charge[char_cap_col].diff()
        cycle_df_charge["dQ/dV"] = cycle_df_charge["Charge_dQ"] / cycle_df_charge["dV"]
        cycle_df_charge[["dQ/dV", "dV", "Charge_dQ"]] = cycle_df_charge[
            ["dQ/dV", "dV", "Charge_dQ"]
        ].fillna(0)
        cycle_df_charge = cycle_df_charge[cycle_df_charge["dQ/dV"] >= 0]

        cycle_df_discharge = cycle_df[cycle_df[curr_col] <= 0]
        cycle_df_discharge["Discharge_dQ"] = cycle_df_discharge[dis_cap_col].diff()
        cycle_df_discharge["dQ/dV"] = (
            cycle_df_discharge["Discharge_dQ"] / cycle_df_discharge["dV"]
        )
        cycle_df_discharge[["dQ/dV", "dV", "Discharge_dQ"]] = cycle_df_discharge[
            ["dQ/dV", "dV", "Discharge_dQ"]
        ].fillna(0)
        cycle_df_discharge = cycle_df_discharge[cycle_df_discharge["dQ/dV"] <= 0]

        print(cycle_df_charge.index, cycle_df_discharge.index)

        cycle_df = pd.concat((cycle_df_charge, cycle_df_discharge), sort=True)
        return cycle_df
    except Exception as ex:
        print(f"ex:{ex}")


def drop_inf_nan_dqdv(cycle_df_dv, datatype):
    """Drop rows where dv=0 (or about 0) in a dataframe that has
    already had dv calculated. Then recalculate dv and calculate dq/dv"""
    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)
    assert "dV" in cycle_df_dv.columns
    assert "dQ/dV" in cycle_df_dv.columns
    assert curr_col in cycle_df_dv.columns

    cycle_df_dv = cycle_df_dv.reset_index(drop=True)

    cycle_df_dv = cycle_df_dv.replace([np.inf, -np.inf], np.nan)
    cycle_df_dv = cycle_df_dv.dropna(subset=["dQ/dV"])

    cycle_df_dv = cycle_df_dv.reset_index(drop=True)
    return cycle_df_dv


def sep_char_dis(df_dqdv, datatype):
    """Takes a dataframe of one cycle with calculated dq/dv and
    separates into charge and discharge differential capacity curves"""
    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)
    assert "dQ/dV" in df_dqdv.columns
    # 01-28-2020: switched from splitting by current col here to splitting by
    # dV.
    switch_cd_index = np.where(np.diff(np.sign(df_dqdv["dV"])))
    if len(switch_cd_index[0]) > 1:
        split = min(switch_cd_index[0], key=lambda x: abs(x - (len(df_dqdv) / 2)))
        # this chooses the split point which is closest to the halfway point of
        # datapoints in the dataframe
    elif len(switch_cd_index[0]) > 0:
        split = int(switch_cd_index[0])
    else:
        split = 0
    firstpart = df_dqdv[: split + 1]
    secondpart = df_dqdv[split + 1 :]
    if firstpart["dQ/dV"].mean() < secondpart["dQ/dV"].mean():
        # then the first part has more negative values
        discharge = firstpart
        charge = secondpart
    else:
        charge = firstpart
        discharge = secondpart

    charge = charge.reset_index(drop=True)
    discharge = discharge.reset_index(drop=True)
    return charge, discharge


def clean_charge_discharge_separately(cd_df, datatype):
    (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    ) = col_variables(datatype)
    cd_df["dV"].iloc[1:] = cd_df[volt_col].diff().iloc[1:]
    num_negs_dv = len(cd_df.loc[(cd_df["dV"] < 0)])
    num_pos_dv = len(cd_df.loc[(cd_df["dV"] > 0)])
    while (num_negs_dv != 0) and (num_pos_dv != 0):
        num_negs = len(cd_df.loc[(cd_df["dQ/dV"] != 0) & (cd_df["dV"] < 0)])
        num_pos = len(cd_df.loc[(cd_df["dQ/dV"] != 0) & (cd_df["dV"] > 0)])
        if num_negs > num_pos:
            cd_df = cd_df.loc[(cd_df["dV"] < 0)]
        else:
            cd_df = cd_df.loc[(cd_df["dV"] >= 0)]
        cd_df["dV"].iloc[1:] = cd_df[volt_col].diff().iloc[1:]
        num_negs_dv = len(cd_df.loc[(cd_df["dV"] < 0)])
        num_pos_dv = len(cd_df.loc[(cd_df["dV"] > 0)])
    cd_df = cd_df.reset_index(drop=True)
    return cd_df


def my_savgolay(dataframe, windowlength, polyorder):
    """Takes battery dataframe with a dQ/dV column and applies a
    sav_golay filter to it, returning the dataframe with a new
    column called Smoothed_dQ/dV"""
    assert not windowlength % 2 == 0
    assert polyorder < windowlength
    unfilt = pd.concat([dataframe["dQ/dV"]])
    unfiltar = unfilt.values
    dataframe["Smoothed_dQ/dV"] = scipy.signal.savgol_filter(
        unfiltar, windowlength, polyorder
    )
    return dataframe


def col_variables(datatype):
    """This function provides a key for column names of the two
    most widely used battery data collection instruments, Arbin, MACCOR, and FE-Neware
    """
    assert datatype == "ARBIN" or datatype == "MACCOR" or datatype == "FE-Neware"

    if datatype == "ARBIN":
        cycle_ind_col = "Cycle_Index"
        data_point_col = "Data_Point"
        volt_col = "Voltage(V)"
        curr_col = "Current(A)"
        dis_cap_col = "Discharge_Capacity(Ah)"
        char_cap_col = "Charge_Capacity(Ah)"
        charge_or_discharge = "Step_Index"

    elif datatype == "MACCOR":
        cycle_ind_col = "Cycle_Index"
        data_point_col = "Rec"
        volt_col = "Voltage(V)"
        curr_col = "Current(A)"
        dis_cap_col = "Cap(Ah)"
        char_cap_col = "Cap(Ah)"
        charge_or_discharge = "Md"

    elif datatype == "FE-Neware":
        cycle_ind_col = "ExternalCycleNumber"
        data_point_col = "ExternalRecordId"
        volt_col = "Voltage"
        curr_col = "Amperage"
        dis_cap_col = "RawCapacityValueDischarge"
        char_cap_col = "RawCapacityValueCharge"
        charge_or_discharge = "State"

    else:
        return None
    return (
        cycle_ind_col,
        data_point_col,
        volt_col,
        curr_col,
        dis_cap_col,
        char_cap_col,
        charge_or_discharge,
    )
