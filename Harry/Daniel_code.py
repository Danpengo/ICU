import pandas as pd
import numpy as np
from os.path import exists
import os
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


general_desc = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]
ts_desc = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
           "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
           "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
           "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
           "TroponinI", "TroponinT", "Urine", "WBC"]  # weight as time series ignored
ts_full = []
for each in ts_desc:
    for end in [".count", ".min", ".mean", ".median", ".max", ".first", ".last"]:
        new = each + end
        ts_full.append(new)

variables = general_desc + ts_full

ICU_df = pd.DataFrame(columns=variables)
all_directory = os.getcwd()
directories = ['/set-a', '/set-b', '/set-c']


for h in range(len(directories)):

    only_files = [f for f in listdir(all_directory + directories[h]) if isfile(join(all_directory + directories[h], f))]
    os.chdir(all_directory + directories[h])

    for filename in range(len(only_files)):
        current_patient = []
        f = only_files[filename]
        print(f)
        convert = pd.read_csv(str(f))
        for desc in general_desc:#non time series
            temp = convert.loc[convert['Parameter'] == desc,'Value']
            if desc != "Weight":
                current_patient.append(float(temp))
            else:
                temp = convert.loc[convert['Parameter'] == desc,['Value']]
                current_patient.append(float(temp['Value'].iat[0]))
        for desc in ts_desc:#time series variables
            temp = convert.loc[convert['Parameter'] == desc,'Value']
            current_patient.append(int(len(temp.index)))#time series counts
            if len(temp.index) != 0:
                current_patient.append(float(temp.min()))#time series min
            else:
                current_patient.append(-1)
            if len(temp.index) != 0:
                current_patient.append(float(temp.mean()))#time series mean
            else:
                current_patient.append(-1)
            if len(temp.index) != 0:
                current_patient.append(float(temp.median()))#time series median
            else:
                current_patient.append(-1)
            if len(temp.index) != 0:
                current_patient.append(float(temp.max()))#time series max
            else:
                current_patient.append(-1)
            temp = convert.loc[convert['Parameter'] == desc,['Value']]
            if int(len(temp.index)) != 0:
                current_patient.append(float(temp['Value'].iat[0]))#time series first
                current_patient.append(float(temp['Value'].iat[-1]))#time series last
            else:
                current_patient.append(-1)
                current_patient.append(-1)
        ICU_df.loc[len(ICU_df.index)] = current_patient

print(ICU_df)

ICU_df.to_csv('ICU_dataset.csv', index=False)


ICU_df2 = pd.read_csv('ICU_dataset.csv')

HeightEstimate = ICU_df2[["Age","Gender","Height"]]
HeightEstimate = HeightEstimate.replace(-1,np.NaN)

imputer = IterativeImputer(estimator=RandomForestRegressor())
imputer.fit(HeightEstimate)

height = pd.DataFrame(imputer.transform(HeightEstimate))
height = height.round(1)

ICU_df2['Height'] = height[2]

WeightEstimate = ICU_df2[["Age","Gender","Height","Weight"]]
WeightEstimate = WeightEstimate.replace(-1,np.NaN)
imputer = IterativeImputer(estimator=RandomForestRegressor())
imputer.fit(WeightEstimate)

weight = pd.DataFrame(imputer.transform(WeightEstimate))
weight = weight.round(1)

ICU_df2['Weight'] = weight[3]

outcome_files = ['Outcomes-a.txt', 'Outcomes-b.txt', 'Outcomes-c.txt']

ICU_df2['In-hospital death'] = ""

count = 0
for i in range(len(outcome_files)):
    ICU_deathstats = pd.read_csv(outcome_files[i])
    for j in range(len(ICU_deathstats)):
        print(ICU_deathstats['In-hospital_death'][j])
        ICU_df2['In-hospital death'][count] = ICU_deathstats['In-hospital_death'][j]
        count += 1

ICU_df2.to_csv('ICU_dataset_death.csv', index=False)

