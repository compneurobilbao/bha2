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


def matrix_fusion(g, fcm, scm):
    cc = np.multiply(((g * abs(fcm)) + ((1 - g) * scm)), np.sign(fcm))
    cc_dist = pdist(cc, "cosine")
    W = cc_dist / max(cc_dist)
    return W


def tree_modules(W, num_clust):
    Z = linkage(W, "weighted")
    T = fcluster(Z, num_clust, criterion="maxclust")
    return T


def level_dictionary(T):
    lvl = T.max()
    l_dict = {}
    for i in range(1, np.max(T) + 1):
        rois_in_clust = np.where(T == i)[0]
        desc = "lvl_" + str(lvl) + "_mod_" + str(i)
        l_dict[desc] = rois_in_clust.tolist()
    return l_dict


def level_connectivity(fc, sc, T):
    lvl = T.max()
    level_features = pd.DataFrame()

    for i in range(1, np.max(T) + 1):
        rois_in_clust = np.where(T == i)[0]
        ext_rois = np.setdiff1d(np.array([i for i in range(len(T))]), rois_in_clust)

        if len(rois_in_clust) > 1:
            desc = "lvl_" + str(lvl) + "_mod_" + str(i)

            fc_int = fc.iloc[rois_in_clust][rois_in_clust].to_numpy().mean(dtype=float)
            fc_out = fc.iloc[rois_in_clust][ext_rois].to_numpy().mean(dtype=float)
            sc_int = (
                sc.iloc[rois_in_clust][rois_in_clust].to_numpy().sum(dtype=float)
            ) / len(rois_in_clust)
            sc_out = (
                sc.iloc[rois_in_clust][ext_rois].to_numpy().sum(dtype=float)
            ) / len(rois_in_clust)

            features = pd.DataFrame(
                {
                    "FCINT_" + desc: fc_int,
                    "FCEXT_" + desc: fc_out,
                    "SCINT_" + desc: sc_int,
                    "SCEXT_" + desc: sc_out,
                },
                index=[0],
            )

            level_features = pd.concat([level_features, features], axis=1)

    return level_features


def tree_connectivity(init_level, end_level, W, sc, fc):
    t_features = pd.DataFrame()
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        l_features = level_connectivity(fc, sc, T)
        t_features = pd.concat([t_features, l_features], axis=1)

    return t_features


def tree_dictionary(init_level, end_level, W):
    t_dict = {}
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        t_dict.update(level_dictionary(T))
    return t_dict
