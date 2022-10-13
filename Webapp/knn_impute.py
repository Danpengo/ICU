import pandas as pd
import numpy as np
from numpy import isnan
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def knn_impute_data(dataset):
    
    """
    The following algorithm imputes missing values in the given dataframe.
    
    parameter dataset: A pandas dataframe
    precondition: The pandas dataframe must be formatted correctly
    return: an imputed dataset
    time_complexity: time complexity is dependant on number of patients, number of predictors,
                     the number of missing values, and the number of outliers within predictors.
    """

    drop_columns = ["TroponinI.count", "TroponinI.min", "TroponinI.mean", "TroponinI.median",
                    "TroponinI.max", "TroponinI.first", "TroponinI.last", "TroponinT.count", "TroponinT.min",
                    "TroponinT.mean", "TroponinT.median", "TroponinT.max", "TroponinT.first", "TroponinT.last",
                    "Cholesterol.count", "Cholesterol.min", "Cholesterol.mean", "Cholesterol.median",
                    "Cholesterol.max", "Cholesterol.first", "Cholesterol.last"]

    dataset.drop(drop_columns, axis=1, inplace=True)

    dataset[dataset.eq(-1)] = np.nan

    pd.set_option('display.max_rows', None)

    dataset = dataset[~dataset['Gender'].isnull()]

    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns)

    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df)), columns=df.columns)

    return df
