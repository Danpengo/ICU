import numpy
from pandas import *
import numpy as np
import os

multiple_values = ["Gender", "Age", "ICUType", "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN",
                   "Cholesterol", "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
                   "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP", "NIMAP",
                   "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate", "SaO2", "SysABP",
                   "Temp", "TroponinI", "TroponinT", "Urine", "WBC", "Weight"]

hours = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
         25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

d = DataFrame('', index=np.arange(48 * 12000), columns=multiple_values)


def collect_data(directory, df):

    list_of_files = os.listdir(directory)
    list_of_files.sort()
    count_of_files = 0

    for filename in list_of_files:
        f = directory + "/" + filename
        data = read_csv(f)

        for i in range(len(data)):
            if data['Parameter'][i] == "RecordID" or data['Parameter'][i] == "Height":
                continue
            df[data['Parameter'][i]][count_of_files + int(data['Time'][i][0:2])] = data['Value'][i]

        for i in range(len(multiple_values)):
            value_found = False
            missing_before_found = []
            for j in range(count_of_files, count_of_files + 48):
                if type(df[multiple_values[i]][j]) is numpy.float64:
                    if df[multiple_values[i]][j] == -1:
                        missing_before_found.append(j)
                        df[multiple_values[i]][j] = ""
                    else:
                        if value_found is False:
                            first_value = df[multiple_values[i]][j]
                        most_recent_value = df[multiple_values[i]][j]
                        value_found = True
                else:
                    if value_found is True:
                        df[multiple_values[i]][j] = most_recent_value
                    else:
                        missing_before_found.append(j)
            if len(missing_before_found) > 0:
                if value_found is True:
                    for h in range(len(missing_before_found)):
                        df[multiple_values[i]][missing_before_found[h]] = first_value

        print(f)
        count_of_files += 48

    return df


all_directory = os.getcwd()
dir = all_directory + '/set-a'
final_a = collect_data(dir, d)

dir = all_directory + '/set-b'
final_b = collect_data(dir, d)

dir = all_directory + '/set-c'
final_c = collect_data(dir, d)

df_final = concat([final_a, final_b, final_c])
df_final = df_final.reset_index()

df_final.to_csv('full_dataset.csv', index=False)
