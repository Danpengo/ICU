import pandas as pd
import numpy as np
from os.path import exists
import os
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def collect_data(path_of_folder, path_of_outcomes):

    general_desc = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]

    ts_desc = ["Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
            "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR", "K",
            "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP", "NISysABP",
            "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP", "Temp",
            "TroponinI", "TroponinT", "Urine", "WBC"]

    ts_full = []
    for each in ts_desc:
        for end in [".count", ".min", ".mean", ".median", ".max", ".first", ".last"]:
            new = each + end
            ts_full.append(new)

    variables = general_desc + ts_full

    ICU_df = pd.DataFrame(columns=variables)
 
    only_files = [f for f in listdir(path_of_folder) if isfile(join(path_of_folder, f))]

    og_directory = os.getcwd()

    os.chdir(og_directory + path_of_folder)

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


    HeightEstimate = ICU_df[["Age","Gender","Height"]]
    HeightEstimate = HeightEstimate.replace(-1,np.NaN)

    imputer = IterativeImputer(estimator=RandomForestRegressor())
    imputer.fit(HeightEstimate)

    height = pd.DataFrame(imputer.transform(HeightEstimate))
    height = height.round(1)

    ICU_df['Height'] = height[2]

    WeightEstimate = ICU_df[["Age","Gender","Height","Weight"]]
    WeightEstimate = WeightEstimate.replace(-1,np.NaN)
    imputer = IterativeImputer(estimator=RandomForestRegressor())
    imputer.fit(WeightEstimate)

    weight = pd.DataFrame(imputer.transform(WeightEstimate))
    weight = weight.round(1)

    ICU_df['Weight'] = weight[3]

    ICU_df['In-hospital death'] = ""

    count = 0

    os.chdir(og_directory + path_of_outcomes)

    ICU_deathstats = pd.read_csv('test_set_outcomes.txt')

    for j in range(len(ICU_deathstats)):
        ICU_df['In-hospital death'][count] = ICU_deathstats['In-hospital_death'][j]
        count += 1

    return ICU_df
