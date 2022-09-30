"""
Main code for BHA 2.0 project
"""

from tree_features import connectome_average, generate_population_features
import json

subject_list = '/project/subjects.txt'
fc_all, sc_all = connectome_average(subject_list)
fcm = np.median(fc_all, axis=0)
scm = np.median(sc_all, axis=0)

#Generate dataframes with module connectivities
gamma = np.arange(0, 1.1, 0.1)
for g in gamma:
    Xdf, Xdesc = generate_population_features(subject_list, g, 10, 20, scm, fcm)
    Xdf.to_csv('/project/df_connectomes/df_gamma_' + str(g) + '.csv', index = False)
    json.dump(battery_corrections, open('/project/modules/gamma_' + str(g) + '_module_desc.json', 'w'))








