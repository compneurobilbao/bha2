import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import os
import scipy.stats as stats
import sys

behavioural_type = sys.argv[1]
testName = sys.argv[2]
columnsNoScore = np.array(sys.argv[3])
inverse_columns = sys.argv[4]

behavioural_path = '/project/behaviour'
listofSubjects = '/project/subjects.txt'

battery_path = os.path.join(behavioural_path, behavioural_type, testName)
df = pd.read_csv(battery_path + '/' + testName + '.csv')
df=df.rename(columns = {'Unnamed: 0':'labels'})
df_sorted = df.sort_values("labels")

s_list = np.genfromtxt(listofSubjects, dtype = "str")
df_young = df_sorted.loc[df['labels'].isin(s_list)]
columns2remove = np.concatenate((['labels'], columnsNoScore))
df_toImpute = df_young.drop(columns=columns2remove)
np.where(pd.isnull(df_toImpute))

tree_est = ExtraTreesRegressor(n_estimators=10, random_state=0)
imp_mean = IterativeImputer(random_state=0, estimator=tree_est)
X = pd.DataFrame(imp_mean.fit_transform(df_toImpute))
X.columns = df_toImpute.columns

X[inverse_columns] = X[inverse_columns] * -1

zstuffed = np.sum(stats.zscore(X), axis=1)/np.sqrt(len(X.columns))
np.savetxt(battery_path + '/' + testName + '_zstuffed.txt', zstuffed)