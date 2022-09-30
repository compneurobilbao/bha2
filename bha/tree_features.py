"""
Functions for calculate connectivity features based on dedrogram tree
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


def connectome_average(fc_all, sc_all):
    fcm = np.median(fc_all, axis=0) - np.diag(np.diag(np.median(fc_all, axis=0)))
    scm = np.median(
        np.array([np.log10(sc + 1) / (np.log10(sc + 1)).max() for sc in sc_all]), axis=0
    )
    return fcm, scm


def tree_modules(g, fcm, scm, num_clust):
    cc = np.multiply(((g * abs(fcm)) + ((1 - g) * scm)), np.sign(fcm))
    cc_dist = pdist(cc, "cosine")
    W = cc_dist / max(cc_dist)
    Z = linkage(W, "weighted")
    T = fcluster(Z, num_clust, criterion="maxclust")
    return T


def level_dictionary(T):
    lvl = T.max()
    t_dict = {}
    for i in range(1, np.max(T) + 1):
        rois_in_clust = np.where(T == i)[0]
        desc = "lvl_" + str(lvl) + "_mod_" + str(i)
        t_dict[desc] = list(rois_in_clust)
    return t_dict


def compute_connectivity(fc, sc, T):
    lvl = T.max()
    features = np.array([])
    feat_names = np.array([])

    for i in range(1, np.max(T) + 1):
        rois_in_clust = np.where(T == i)[0]
        ext_rois = np.setdiff1d(np.array([i for i in range(len(T))]), rois_in_clust)

        if len(rois_in_clust) > 1:
            desc = "lvl_" + str(lvl) + "_mod_" + str(i)

            fc_int = fc.iloc[rois_in_clust][rois_in_clust].to_numpy().mean()
            fc_out = fc.iloc[rois_in_clust][ext_rois].to_numpy().mean()
            sc_int = (sc.iloc[rois_in_clust][rois_in_clust].to_numpy().sum()) / len(
                rois_in_clust
            )
            sc_out = (sc.iloc[rois_in_clust][ext_rois].to_numpy().sum()) / len(
                rois_in_clust
            )

            features = pd.DataFrame(
                np.concatenate((features, np.array([fc_int, fc_out, sc_int, sc_out])))
            )
            feat_names = np.concatenate(
                (
                    feat_names,
                    np.array(
                        [
                            "FCINT_" + desc,
                            "FCEXT_" + desc,
                            "SCINT_" + desc,
                            "SCEXT_" + desc,
                        ]
                    ),
                )
            )
            features.columns = feat_names
    return features


# def generate_population_features(slist, g, num_clust_init, num_clust_end, scm, fcm):
#     cc = np.multiply(((g * abs(fcm)) + ((1 - g) * scm)), np.sign(fcm))
#     cc_dist = pdist(cc, "cosine")
#     W = cc_dist / max(cc_dist)
#     Z = linkage(W, "weighted")

#     X_features = slist.reshape(len(slist), 1)
#     X_names = np.array(["label"])
#     X_desc_dict = {}
#     for num_clust in range(num_clust_init, num_clust_end + 1, 1):
#         T = fcluster(Z, num_clust, criterion="maxclust")
#         pfeatures = []

#         for i, sub in enumerate(slist):
#             print(
#                 "gamma = "
#                 + str(g)
#                 + ", lvl = "
#                 + str(num_clust)
#                 + ", snumber = "
#                 + str(i)
#             )
#             sc = np.array(
#                 pd.read_csv(
#                     "sc_matrices/" + sub + "_anat_probabilistic_connectome.csv",
#                     delimiter=" ",
#                     header=None,
#                 )
#             )
#             fc = np.corrcoef(
#                 np.transpose(np.genfromtxt("timeseries/ts_" + sub + ".txt"))
#             )
#             scmod = np.log10(sc + 1)
#             subfeatures, subfeat_names, feat_dict = compute_connectivity(
#                 fc, scmod / scmod.max(), T, num_clust
#             )
#             pfeatures.append(subfeatures)

#         X_features = np.hstack((X_features, np.array(pfeatures)))
#         X_names = np.hstack((X_names, np.array(subfeat_names)))
#         X_desc_dict.update(feat_dict)

#     Xdf = pd.DataFrame(X_features)
#     Xdf.columns = X_names
#     Xdf = Xdf.loc[:, ~Xdf.apply(lambda x: x.duplicated(), axis=1).all()].copy()
#     return Xdf, X_desc_dict
