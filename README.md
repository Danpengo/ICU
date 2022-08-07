# ICU
ICU Mortality Predictions Project

## time_series_data.csv 

time_series_data.csv utilises 34 time-series variables for patients. The input of each patient consists of two general descriptors (Age + ICU Type) and thirty-four time-series variables. I recorded 34 time-series variables about the vital signs of the patients with a one hour time interval, so each patient has exactly 48 records for each variable.

I dealt with missing values by creating a smaller dataset that takes the average value (or median value for categorical variables) for each patient. This created a dataset of size 4000 (one row for each patient). I then performed knn imputation and used the values imputed for missing variables in the larger time series dataset. 

The reason I chose to do this instead of performing knn imputation on the larger time series dataset was for two reasons:
1. The dataset was too large to perform knn imputation (too much computational complexity)
2. I didn't want to create any time series patterns from knn imputation that may not represent reality.
