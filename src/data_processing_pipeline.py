import pandas as pd
import numpy as np
import time

import os
from pathlib import Path

import torch


def pre_merge_clean_holidays(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["description"] = df["description"].str.replace(r'\+\d+$', '', regex=True) #removes +n from specific holidays
    df["description"] = df["description"].str.replace(r'\-\d+$', '', regex=True) #removes -n from specific holidays

    new_holidays_col_names = {
        "description":"holiday_description",
        "type":"holiday_type"
    }
    df.rename(columns=new_holidays_col_names, inplace=True)

    return df


def pre_merge_clean_oil(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    df = df.interpolate(method="linear")

    df.rename(columns={"dcoilwtico":"dol_per_barrel"}, inplace=True)
    
    return df

def pre_merge_clean_stores(df):
    new_stores_col_names = {
        "type":"store_type",
        "cluster":"store_cluster",
        "transactions":"store_transactions_per_day"
    }
    df.rename(columns=new_stores_col_names, inplace=True)

    return df

def post_merge_cleaning(df):
    ##interpolate oil  
    df = df.interpolate(method="linear")

    df["date"] = pd.to_datetime(df["date"]) #re-instantiatind a pliable format

    ##lag feature creation

    #lag feature creation
    deduplicated = df.groupby(["date", "store_nbr"], as_index=False)["transactions"].first() #2023-01-01 actually only appears once here for one store...probably have to omit

    deduplicated["rolling_30day_avg_traffic"] = deduplicated.groupby("store_nbr")["transactions"].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    deduplicated["rolling_15day_avg_traffic"] = deduplicated.groupby("store_nbr")["transactions"].transform(lambda x: x.rolling(window=15, min_periods=1).mean())


    intra_col_merge_keys = ["date", "store_nbr", "rolling_30day_avg_traffic", "rolling_15day_avg_traffic"]
    df = df.merge(deduplicated[intra_col_merge_keys], on=[intra_col_merge_keys[0], intra_col_merge_keys[1]], how="left")
        ##include the drop of the NaNs set from the lag feature creation

    ##day of week

    def make_train_test(df):
        ##max date subtracted by 15 and output as test
        pass

    return df



def run_base_df():
    transx = pd.read_csv("../data/transactions.csv")
    stores = pd.read_csv("../data/stores.csv")
    oil = pd.read_csv("../data/oil.csv")
    holidays = pd.read_csv("../data/holidays_events.csv")

    training = pd.read_csv("../data/train.csv")
    # testing = pd.read_csv("../data/test.csv")
    # sample = pd.read_csv("../data/sample_submission.csv")

    dated_dfs = [transx, oil, holidays, training] #oil, holidays,
    # testing["date"] = pd.to_datetime(testing["date"]) #this is to show how we'll tag our testing df with all the pre-processing we need {A function will be the result of housing all the cleaning}

    for df in dated_dfs:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # print(type(df["date"][0]))

    holidays = pre_merge_clean_holidays(holidays)
    oil = pre_merge_clean_oil(oil)
    stores = pre_merge_clean_stores(stores)

    #Training
    print("Training")
    base_df = pd.merge(training, stores, on="store_nbr", how="inner")
    base_df = base_df.set_index("id")
    base_df = pd.merge(base_df, transx, on=["date", "store_nbr"], how="inner")
    base_df = pd.merge(base_df, oil, on="date", how="left")
    base_df = pd.merge(base_df, holidays, on="date", how="left")

    base_df = post_merge_cleaning(base_df)

    return base_df