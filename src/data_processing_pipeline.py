#### IMPORTS ####
import pandas as pd
import numpy as np
import time

import os
from pathlib import Path


from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.stattools import adfuller


import torch

#### HELPER FUNCTIONS ####
def _pre_merge_clean_holidays(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date

    df["description"] = df["description"].str.replace(r'\+\d+$', '', regex=True) #removes +n from specific holidays
    df["description"] = df["description"].str.replace(r'\-\d+$', '', regex=True) #removes -n from specific holidays

    new_holidays_col_names = {
        "description":"holiday_description",
        "type":"holiday_type"
    }
    df.rename(columns=new_holidays_col_names, inplace=True)

    return df


def _pre_merge_clean_oil(df):
    df["date"] = pd.to_datetime(df["date"]).dt.date
    
    df = df.interpolate(method="linear")

    df.rename(columns={"dcoilwtico":"dol_per_barrel"}, inplace=True)
    
    return df

def _pre_merge_clean_stores(df):
    new_stores_col_names = {
        "type":"store_type",
        "cluster":"store_cluster",
        "transactions":"store_transactions_per_day"
    }
    df.rename(columns=new_stores_col_names, inplace=True)

    return df

def _post_merge_cleaning(df):
    ##interpolate oil  
    df = df.interpolate(method="linear")
    # df["dol_per_barrel"] = df["dol_per_barrel"].interpolate(method="linear").bfill()
    print(f"Post merge cleaning with 2nd interpolation of oil pricing...")
    print(f"Corresponding null count of df is: {df.isnull().sum()}")

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





#### BASE DATAFRAMEFUNCTIONS ####
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

    #Convert date to datetime in eligible dfs
    for df in dated_dfs:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # print(type(df["date"][0]))


    holidays = _pre_merge_clean_holidays(holidays)
    oil = _pre_merge_clean_oil(oil)
    stores = _pre_merge_clean_stores(stores)

    #Training
    print("Training")
    base_df = pd.merge(training, stores, on="store_nbr", how="inner")
    base_df = base_df.set_index("id")
    base_df = pd.merge(base_df, transx, on=["date", "store_nbr"], how="inner")
    base_df = pd.merge(base_df, oil, on="date", how="left")
    base_df = pd.merge(base_df, holidays, on="date", how="left")

    base_df = _post_merge_cleaning(base_df)

    return base_df


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

    #Convert date to datetime in eligible dfs
    for df in dated_dfs:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        # print(type(df["date"][0]))


    holidays = _pre_merge_clean_holidays(holidays)
    oil = _pre_merge_clean_oil(oil)
    stores = _pre_merge_clean_stores(stores)

    #Training
    print("Training")
    base_df = pd.merge(training, stores, on="store_nbr", how="inner")
    base_df = base_df.set_index("id")
    base_df = pd.merge(base_df, transx, on=["date", "store_nbr"], how="inner")
    base_df = pd.merge(base_df, oil, on="date", how="left")
    base_df = pd.merge(base_df, holidays, on="date", how="left")

    base_df = _post_merge_cleaning(base_df)

    return base_df







#### MODELINING FUNCTIONS ####
def antiquated_single_model_pre_process(base_df, model_filter=[45, "GROCERY I"], first_n_dates=30, last_n_dates=15):
    
    ##Apply filter
    base_df = base_df[
        (base_df["store_nbr"] == model_filter[0]) &
        (base_df["family"] == model_filter[1])
    ]

    ##Assert date_index
    base_df["date_index"] = base_df["date"]
    base_df.set_index("date_index", inplace=True)

    ##One hot cols
    one_hot_columns = [
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type",
        "locale",
        "locale_name",
        "holiday_description",
        "transferred"
    ]
    one_hot_prep_df = base_df[one_hot_columns]
    one_hot_prep_df = one_hot_prep_df.fillna("no_holiday")
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(base_df[one_hot_columns])
    encoded_dense = encoded.toarray()
    encoded_df = pd.DataFrame(encoded_dense, columns=encoder.get_feature_names_out(one_hot_columns), index=base_df.index)

    ##Day of Week encoding
    base_df["date"] = pd.to_datetime(base_df["date"]) #redundancy due to bug?
    base_df["day_of_week"] = base_df["date"].dt.day_name()
    DAY_OF_WEEK_HIERARCHY_DICT = {
        "Sunday":1, 
        "Monday":2,
        "Tuesday":3,
        "Wednesday":4,
        "Thursday":5,
        "Friday":6,
        "Saturday":7
    }
    niche_df = base_df.copy()
    niche_df = niche_df[["day_of_week"]]
    niche_df["day_of_week_hierarchy_encoded"] = niche_df["day_of_week"].map(DAY_OF_WEEK_HIERARCHY_DICT)
    niche_df["day_of_week_sine"] = np.sin(2 * np.pi * niche_df["day_of_week_hierarchy_encoded"] / 7)
    niche_df["day_of_week_cosine"] = np.cos(2 * np.pi * niche_df["day_of_week_hierarchy_encoded"] / 7)
    niche_df.drop(columns=["day_of_week", "day_of_week_hierarchy_encoded"], inplace=True)


    ##Model_df generation
    model_df = base_df.copy()
    model_df = model_df[["date", "family", "sales", "dol_per_barrel", "rolling_15day_avg_traffic", "rolling_30day_avg_traffic", "store_nbr"]]

    #Weird redundant cleaning, but it works (this step will repeat)
    model_df = model_df[~model_df.index.duplicated(keep="first")]
    
    #Concat
    model_df = pd.concat([model_df, niche_df, encoded_df], axis=1)

    #Weird redundant cleaning, but it works (this step will repeat)
    model_df = model_df[~model_df.index.duplicated(keep="first")]
    model_df.dropna(inplace=True)
    model_df.sort_values(by=["date", "store_nbr", "family"], ascending=True)

    #n_date cutting (train, test, split)
    first_n_dates = model_df["date"].drop_duplicates().head(first_n_dates) 
    last_n_dates = model_df["date"].drop_duplicates().tail(last_n_dates) 
    model_train = model_df[
        (~model_df["date"].isin(first_n_dates)) &
        (~model_df["date"].isin(last_n_dates))
    ]
    model_test = model_df[
        (model_df["date"].isin(last_n_dates))
    ]
    X_train = model_train.drop(columns=["sales", "family", "date"])
    y_train = model_train["sales"]
    X_test = model_test.drop(columns=["sales", "family", "date"])
    y_test = model_test["sales"]

    #Weird redundant cleaning, but it works (this step will repeat)
    X_train.dropna(inplace=True)
    y_train.dropna(inplace=True)
    X_test.dropna(inplace=True)
    y_test.dropna(inplace=True)

    #Set index frequency encoding
    X_train = X_train.asfreq("D")
    y_train = y_train.asfreq("D")

    X_test = X_test.asfreq("D")
    y_test = y_test.asfreq("D")

    #Weird redundant cleaning, but it works (this step will repeat)
    X_train.drop_duplicates(keep="first")
    y_train.drop_duplicates(keep="first")
    X_test.drop_duplicates(keep="first")
    y_test.drop_duplicates(keep="first")
    X_train.fillna(0, inplace=True)
    y_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    y_test.fillna(0, inplace=True)

    return X_train, y_train, X_test, y_test




def debug_single_model_pre_process(base_df, model_filter=[45, "GROCERY I"], first_n_dates=30, last_n_dates=15):
    
    ##Apply filter
    base_df = base_df[
        (base_df["store_nbr"] == model_filter[0]) &
        (base_df["family"] == model_filter[1])
    ]
    print(f"1) Corresponding null count of base_df in modeling is: {base_df.isnull().sum().sum()}")

    ##Assert date_index
    base_df["date_index"] = base_df["date"]
    base_df.set_index("date_index", inplace=True)
    print(f"2) Corresponding null count of base_df in modeling is: {base_df.isnull().sum().sum()}")

    ##One hot cols
    one_hot_columns = [
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type",
        "locale",
        "locale_name",
        "holiday_description",
        "transferred"
    ]
    one_hot_prep_df = base_df[one_hot_columns]
    one_hot_prep_df = one_hot_prep_df.fillna("no_holiday")
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(base_df[one_hot_columns])
    encoded_dense = encoded.toarray()
    encoded_df = pd.DataFrame(encoded_dense, columns=encoder.get_feature_names_out(one_hot_columns), index=base_df.index)
    print(f"3) Corresponding null count of the encoded_df in modeling is: {encoded_df.isnull().sum().sum()}")

    ##Day of Week encoding
    base_df["date"] = pd.to_datetime(base_df["date"]) #redundancy due to bug?
    base_df["day_of_week"] = base_df["date"].dt.day_name()
    DAY_OF_WEEK_HIERARCHY_DICT = {
        "Sunday":1, 
        "Monday":2,
        "Tuesday":3,
        "Wednesday":4,
        "Thursday":5,
        "Friday":6,
        "Saturday":7
    }
    niche_df = base_df.copy()
    niche_df = niche_df[["day_of_week"]]
    niche_df["day_of_week_hierarchy_encoded"] = niche_df["day_of_week"].map(DAY_OF_WEEK_HIERARCHY_DICT)
    niche_df["day_of_week_sine"] = np.sin(2 * np.pi * niche_df["day_of_week_hierarchy_encoded"] / 7)
    niche_df["day_of_week_cosine"] = np.cos(2 * np.pi * niche_df["day_of_week_hierarchy_encoded"] / 7)
    niche_df.drop(columns=["day_of_week", "day_of_week_hierarchy_encoded"], inplace=True)
    print(f"4) Corresponding null count of the nice_df in modeling is: {niche_df.isnull().sum().sum()}")


    ##Model_df generation
    model_df = base_df.copy()
    model_df = model_df[["date", "family", "sales", "dol_per_barrel", "rolling_15day_avg_traffic", "rolling_30day_avg_traffic", "store_nbr"]]
    print(f"5) Corresponding null count of the model_df in modeling is: {model_df.isnull().sum().sum()}")

    #Weird redundant cleaning, but it works (this step will repeat)
    model_df = model_df[~model_df.index.duplicated(keep="first")]
    print(f"6) Corresponding null count of the model_df in modeling is: {model_df.isnull().sum().sum()}")
    
    #Concat
    model_df = pd.concat([model_df, niche_df, encoded_df], axis=1)
    print(f"7) Corresponding null count of the model_df in modeling is: {model_df.isnull().sum().sum()}")

    #Weird redundant cleaning, but it works (this step will repeat)
    model_df = model_df[~model_df.index.duplicated(keep="first")]
    print(f"8) Corresponding null count of the model_df in modeling is: {model_df.isnull().sum().sum()}")
    model_df.dropna(inplace=True)
    print(f"9) Corresponding null count of the model_df in modeling is: {model_df.isnull().sum().sum()}")
    model_df.sort_values(by=["date", "store_nbr", "family"], ascending=True)

    #n_date cutting (train, test, split)
    first_n_dates = model_df["date"].drop_duplicates().head(first_n_dates) 
    last_n_dates = model_df["date"].drop_duplicates().tail(last_n_dates) 
    model_train = model_df[
        (~model_df["date"].isin(first_n_dates)) &
        (~model_df["date"].isin(last_n_dates))
    ]
    print(f"10) Corresponding null count of the model_df in modeling is: {model_train.isnull().sum().sum()}")
    model_test = model_df[
        (model_df["date"].isin(last_n_dates))
    ]
    print(f"11) Corresponding null count of the model_df in modeling is: {model_test.isnull().sum().sum()}")
    X_train = model_train.drop(columns=["sales", "family", "date"])
    y_train = model_train["sales"]
    X_test = model_test.drop(columns=["sales", "family", "date"])
    y_test = model_test["sales"]

    #Weird redundant cleaning, but it works (this step will repeat)
    X_train.dropna(inplace=True)
    y_train.dropna(inplace=True)
    X_test.dropna(inplace=True)
    y_test.dropna(inplace=True)
    print(f"12) Corresponding null count of the X_train in modeling is: {X_train.isnull().sum().sum()}")
    print(f"13) Corresponding null count of the y_train in modeling is: {y_train.isnull().sum().sum()}")
    print(f"14) Corresponding null count of the X_test in modeling is: {X_test.isnull().sum().sum()}")
    print(f"15) Corresponding null count of the y_test in modeling is: {y_test.isnull().sum().sum()}")

    #Set index frequency encoding
    X_train = X_train.asfreq("D")
    y_train = y_train.asfreq("D")
    print(f"16) Corresponding null count of the X_train in modeling is: {X_train.isnull().sum().sum()}")
    print(f"17) Corresponding null count of the y_train in modeling is: {y_train.isnull().sum().sum()}")

    X_test = X_test.asfreq("D")
    y_test = y_test.asfreq("D")
    print(f"18) Corresponding null count of the X_test in modeling is: {X_test.isnull().sum().sum()}")
    print(f"19) Corresponding null count of the y_test in modeling is: {y_test.isnull().sum().sum()}")

    #Weird redundant cleaning, but it works (this step will repeat)
    X_train.drop_duplicates(keep="first")
    y_train.drop_duplicates(keep="first")
    X_test.drop_duplicates(keep="first")
    y_test.drop_duplicates(keep="first")
    print(f"20) Corresponding null count of the X_train in modeling is: {X_train.isnull().sum().sum()}")
    print(f"21) Corresponding null count of the y_train in modeling is: {y_train.isnull().sum().sum()}")
    print(f"22) Corresponding null count of the X_test in modeling is: {X_test.isnull().sum().sum()}")
    print(f"23) Corresponding null count of the y_test in modeling is: {y_test.isnull().sum().sum()}")
    X_train.fillna(0, inplace=True)
    y_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)
    y_test.fillna(0, inplace=True)
    print(f"24) Corresponding null count of the X_train in modeling is: {X_train.isnull().sum().sum()}")
    print(f"25) Corresponding null count of the y_train in modeling is: {y_train.isnull().sum().sum()}")
    print(f"26) Corresponding null count of the X_test in modeling is: {X_test.isnull().sum().sum()}")
    print(f"27) Corresponding null count of the y_test in modeling is: {y_test.isnull().sum().sum()}")

    return X_train, y_train, X_test, y_test






def sarimax_model_pre_process(base_df, model_filter=[45, "GROCERY I"], first_n_dates=30, last_n_dates=15):
    pass

def lstm_model_pre_process(base_df, model_filter=[45, "GROCERY I"], first_n_dates=30, last_n_dates=15):
    pass



