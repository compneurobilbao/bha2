import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import scipy.stats as stats
import sys
import json

behavioural_type = sys.argv[1]
test_name = sys.argv[2]


behavioural_path = "/home/antoniojm/Documents/projects/BHA_5G/lemon/behaviour"
s_list_path = "/home/antoniojm/Documents/projects/BHA_5G/lemon/subjects_blood.txt"

battery_path = os.path.join(behavioural_path, behavioural_type, test_name)
df = pd.read_csv(battery_path + "/" + test_name + ".csv")
df = df.rename(columns={"Unnamed: 0": "labels"})
df_sorted = df.sort_values("labels")
battery_corrections = json.loads(
    open(battery_path + "/" + test_name + "_corrections.json").read()
)
columns_no_score = battery_corrections["tasks_to_remove"]
inverse_columns = battery_corrections["tasks_to_invert"]

s_list = np.genfromtxt(s_list_path, dtype="str")
df_young = df_sorted.loc[df["labels"].isin(s_list)]
print(np.where(pd.isnull(df_young)))
columns_to_remove = np.concatenate((["labels"], columns_no_score))
df_to_impute = df_young.drop(columns=columns_to_remove)
np.where(pd.isnull(df_to_impute))

tree_est = ExtraTreesRegressor(n_estimators=10, random_state=0)
imp_mean = IterativeImputer(random_state=0, estimator=tree_est)
X = pd.DataFrame(imp_mean.fit_transform(df_to_impute))
X.columns = df_to_impute.columns

X[inverse_columns] = X[inverse_columns] * -1

zstuffed = np.sum(stats.zscore(X), axis=1) / np.sqrt(len(X.columns))
np.savetxt(battery_path + "/" + test_name + "_zstuffed.txt", zstuffed)
