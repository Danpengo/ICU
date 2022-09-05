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

ICU_df = pd.read_csv('ICU_dataset_death.csv')

drop_columns = ["TroponinI.count", "TroponinI.min", "TroponinI.mean", "TroponinI.median",
                "TroponinI.max", "TroponinI.first", "TroponinI.last", "TroponinT.count", "TroponinT.min",
                "TroponinT.mean", "TroponinT.median", "TroponinT.max", "TroponinT.first", "TroponinT.last",
                "Cholesterol.count", "Cholesterol.min", "Cholesterol.mean", "Cholesterol.median",
                "Cholesterol.max", "Cholesterol.first", "Cholesterol.last"]

ICU_df.drop(drop_columns, axis=1, inplace=True)

ICU_df[ICU_df.eq(-1)] = np.nan

pd.set_option('display.max_rows', None)

ICU_df = ICU_df[~ICU_df['Gender'].isnull()]
print(ICU_df.isnull().sum())

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(ICU_df), columns=ICU_df.columns)

imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(scaler.inverse_transform(imputer.fit_transform(df)), columns=df.columns)

df.to_csv('ICU_dataset_death_knnimputed.csv', index=False)

'''
# optional to check for k
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(ICU_df), columns=ICU_df.columns)
predictor = len(df.columns) - 1
data = df.values
ix = [i for i in range(data.shape[1]) if i != predictor]
X, y = data[:, ix], data[:, predictor]
results = list()
strategies = [str(i) for i in [5, 10]]

for s in strategies:
    # create the modeling pipeline
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # store results
    results.append(scores)
    print('>%s %.3f (%.3f)' % (s, mean(scores), std(scores)))
'''

