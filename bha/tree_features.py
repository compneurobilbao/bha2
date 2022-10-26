"""
Functions for calculate connectivity features based on dedrogram tree
"""

import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import nibabel as nib
import json
from numba import njit, jit, prange


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


def get_module_matrix(matrix, rois):
    module_matrix = matrix[rois, :][:, rois]
    return module_matrix


def threshold_based_similarity(fcm, scm, tree):

    module_sim = []
    module_thr = {}
    for levels in tree:
        rois = tree[levels]
        if len(rois) > 1:
            mod_fc = get_module_matrix(fcm, rois)
            mod_sc = get_module_matrix(scm, rois)
            similarities = []
            tresholds = []
            for thr_a in np.arange(0, 1, 0.1):
                for thr_b in np.arange(0, 1, 0.1):
                    thr_fc = np.where(abs(mod_fc) > thr_a, 1, 0)
                    thr_sc = np.where(mod_sc > thr_b, 1, 0)
                    if (thr_fc.sum() + thr_sc.sum()) != 0:
                        thr_sim = (
                            2
                            * np.multiply(thr_fc, thr_sc).sum()
                            / (thr_fc.sum() + thr_sc.sum())
                        )
                        similarities.append(thr_sim)
                        tresholds.append(np.array([thr_a, thr_b]))
            module_sim.append(np.array(similarities).max())
            module_thr[levels] = tresholds[np.array(similarities).argmax()]
    return module_sim, module_thr


def modularity(A, T):
    N = len(T)
    K = np.array(A.sum(axis=0).reshape(1, -1), dtype=np.float64)
    m = K.sum()
    B = A - (K.T * K) / m
    s = np.array([T for i in range(N)], dtype=np.float64)
    Q = B[np.where((s.T - s) == 0)].sum() / m
    return Q


def cross_modularity(fc, sc, g, l):
    W = matrix_fusion(g, fc, sc)
    T = tree_modules(W, l)
    level = level_dictionary(T)

    sims, thrs = threshold_based_similarity(fc, sc, level)
    mod_sc = modularity(sc, T)
    mod_fc = modularity(fc, T)
    crossmod = pow((np.array(sims).mean() * mod_sc * mod_fc), (1 / 3))

    return crossmod, thrs


def level_connectivity(fc, sc, T):
    lvl = T.max()
    level_features = np.array([])
    level_features_names = np.array([])

    for i in range(1, np.max(T) + 1):
        rois_in_clust = np.where(T == i)[0]
        ext_rois = np.setdiff1d(np.array([i for i in range(len(T))]), rois_in_clust)

        if len(rois_in_clust) > 1:
            desc = "lvl_" + str(lvl) + "_mod_" + str(i)

            fc_int = fc[rois_in_clust, :][:, rois_in_clust].mean(dtype=float)
            fc_out = fc[rois_in_clust[:, None], ext_rois].mean(dtype=float)
            sc_int = (sc[rois_in_clust, :][:, rois_in_clust].sum(dtype=float)) / len(
                rois_in_clust
            )
            sc_out = (sc[rois_in_clust[:, None], ext_rois].sum(dtype=float)) / len(
                rois_in_clust
            )
            level_features = np.hstack(
                [level_features, np.array([fc_int, fc_out, sc_int, sc_out])]
            )
            level_features_names = np.hstack(
                [
                    level_features_names,
                    np.array(
                        [
                            "FCINT_" + desc,
                            "FCOUT_" + desc,
                            "SCINT_" + desc,
                            "SCOUT_" + desc,
                        ]
                    ),
                ],
            )

    return level_features, level_features_names


def tree_connectivity(init_level, end_level, W, sc, fc):
    t_features = np.array([])
    t_features_names = np.array([])
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        l_features, l_names = level_connectivity(fc, sc, T)
        t_features = np.hstack([t_features, l_features])
        t_features_names = np.hstack([t_features_names, l_names])

    return t_features, t_features_names


def tree_dictionary(init_level, end_level, W):
    t_dict = {}
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        t_dict.update(level_dictionary(T))
    return t_dict


def get_module_img(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_img = np.where(atlas_data == (np.array(rois) + 1), value, 0).sum(axis=3)
    img = nib.Nifti1Image(module_img, affine=atlas.affine)
    return img


def add_gamma_to_lvl_dict(dict, g):
    oldkeys = list(dict.keys())
    newkeys = [
        s.replace("lvl_", "gamma_" + str(round(g, 2)) + "_lvl_") for s in oldkeys
    ]
    vals = list(dict.values())
    newdictionary = {k: v for k, v in zip(newkeys, vals)}
    return newdictionary
