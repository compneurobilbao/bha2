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
import networkx as nx

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


def tree_dictionary(init_level, end_level, W, tree_class = "reduced"):
    t_dict = {}
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        level_dict = level_dictionary(T)
        for mod in level_dict:
            rois = level_dict[mod]
            if tree_class == "reduced":
                if rois not in list(t_dict.values()):
                    t_dict.update({mod: rois})
            elif tree_class == "full":
                t_dict.update({mod: rois})
            else:
                raise ValueError("tree_class must be 'reduced' or 'full'")
    return t_dict



def get_module_matrix(matrix, rois):
    module_matrix = matrix[rois, :][:, rois]
    return module_matrix


def level_from_tree(tree, level_num):
    level_keys = list(filter(lambda x: x.startswith('lvl_' + str(level_num) + '_'), tree.keys()))
    level = [tree[key] for key in level_keys]
    labels = [key for key in level_keys]
    return level, labels


def T_from_level(level):
    T = np.zeros(max(map(max, level))+1)
    for idx, mod in enumerate(level):
        T[mod] = idx+1
    return T

def adj_matrices_from_level(sc, fc, level, fc_type='pos', thresh=0.2):
    reduced_sc = np.zeros((len(level), len(level)))
    reduced_fc = np.zeros((len(level), len(level)))
    if fc_type == 'abs':
        fc_mod = np.abs(fc)
    elif fc_type == 'sign':
        fc_mod = fc
    elif fc_type == 'pos':
        fc_mod = np.where(fc > 0, fc, 0)
    elif fc_type == 'thr':
        fc_mod = np.where(np.abs(fc) > thresh, fc, 0)
    for i, int_rois in enumerate(level):
        for j, ext_rois in enumerate(level):
            if i == j:
                reduced_sc[i, j] = 0.0
                reduced_fc[i, j] = 0.0
            else:
                sc_out = np.nansum(sc[int_rois][:,ext_rois], dtype=float) / len(int_rois)
                fc_out = np.nanmean(fc_mod[int_rois][:,ext_rois], dtype=float)
                reduced_sc[i, j] = sc_out
                reduced_fc[i, j] = fc_out
    return reduced_sc, reduced_fc


def network_from_level(sc, fc, level, labels):
    reduced_sc, reduced_fc = adj_matrices_from_level(sc, fc, level)
    node_dict = {idx: label for idx, label in enumerate(labels)}
    G_sc = nx.relabel_nodes(nx.from_numpy_matrix(reduced_sc), node_dict, copy=False)
    G_fc = nx.relabel_nodes(nx.from_numpy_matrix(reduced_fc), node_dict, copy=False)
    return G_sc, G_fc


def threshold_based_similarity(fcm, scm, level, labels):

    module_sim = []
    module_thr = {}
    for rois, mod_label in zip(level, labels):
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
            module_thr[mod_label] = tresholds[np.array(similarities).argmax()]
    return module_sim, module_thr


def modularity(A, T):
    N = len(T)
    K = np.array(A.sum(axis=0).reshape(1, -1), dtype=np.float64)
    m = K.sum()
    B = A - (K.T * K) / m
    s = np.array([T for i in range(N)], dtype=np.float64)
    Q = B[np.where((s.T - s) == 0)].sum() / m
    return Q


def cross_modularity(fc, sc, level, labels):
    sims, thrs = threshold_based_similarity(fc, sc, level, labels)
    T = T_from_level(level)
    mod_sc = modularity(sc, T)
    mod_fc = modularity(fc, T)
    crossmod = pow((np.array(sims).mean() * mod_sc * mod_fc), (1 / 3))

    return crossmod, thrs

def network_measures(Graph):
    strength = {node: 0 for node in Graph.nodes()}
    for u, v, weight in Graph.edges(data='weight'):
        strength[u] += weight
        strength[v] += weight
    betweenness = nx.betweenness_centrality(Graph, weight='weight')
    clustering_coef = nx.clustering(Graph, weight='weight')
    pathlength = dict(nx.shortest_path_length(Graph, weight='weight'))
    node_avg_pathlength = {node: 0 for node in Graph.nodes()}
    for node,plengths in pathlength.items():
        node_avg_pathlength[node] = sum(plengths.values())
    n_measures = {key: [strength[key], betweenness[key], 
        clustering_coef[key], node_avg_pathlength[key]] 
        for key in betweenness.keys()}
    return n_measures


def brain_maps_network_measures(tree, sc, fc, atlas, range_of_levels):
    atlas_data = atlas.get_fdata()
    strength_sc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    strength_fc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    betweenness_sc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    betweenness_fc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    c_coeff_sc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    c_coeff_fc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    p_length_sc_vol = np.zeros(atlas_data[:,:,:,0].shape)
    p_length_fc_vol = np.zeros(atlas_data[:,:,:,0].shape)

    for lv in range_of_levels:
        level, labels = level_from_tree(tree, lv)
        sc_level_net, fc_level_net = network_from_level(sc, fc, level, labels)
        n_measures_sc = network_measures(sc_level_net)
        n_measures_fc = network_measures(fc_level_net)
        for rois, mod_label in zip(level, labels):
            module_vol = get_module_vol(atlas, rois)
            strength_sc_vol = strength_sc_vol + module_vol*n_measures_sc[mod_label][0]
            strength_fc_vol = strength_fc_vol + module_vol*n_measures_fc[mod_label][0]
            betweenness_sc_vol = betweenness_sc_vol + module_vol*n_measures_sc[mod_label][1]
            betweenness_fc_vol = betweenness_fc_vol + module_vol*n_measures_fc[mod_label][1]
            c_coeff_sc_vol = c_coeff_sc_vol + module_vol*n_measures_sc[mod_label][2]
            c_coeff_fc_vol = c_coeff_fc_vol + module_vol*n_measures_fc[mod_label][2]
            p_length_sc_vol = p_length_sc_vol + module_vol*n_measures_sc[mod_label][3]
            p_length_fc_vol = p_length_fc_vol + module_vol*n_measures_fc[mod_label][3]

    sc_img_list = [strength_sc_vol, betweenness_sc_vol, c_coeff_sc_vol, p_length_sc_vol]
    fc_img_list = [strength_fc_vol, betweenness_fc_vol, c_coeff_fc_vol, p_length_fc_vol]
    return sc_img_list, fc_img_list
            
   

def module_connectivity(rois, label, sc, fc):
    int_rois = np.array(rois)
    ext_rois = np.setdiff1d(np.array([i for i in range(len(sc))]), int_rois)

    fc_int = np.nanmean(fc[int_rois, :][:, int_rois], dtype=float)
    fc_out = np.nanmean(fc[int_rois[:, None], ext_rois], dtype=float)
    sc_int = np.nansum(sc[int_rois, :][:, int_rois], dtype=float) / len(int_rois)
    sc_out = np.nansum(sc[int_rois[:, None], ext_rois], dtype=float) / len(int_rois)
    module_features = np.array([fc_int, fc_out, sc_int, sc_out])
    module_labels = np.array(
        ["FCINT_" + label, "FCOUT_" + label, "SCINT_" + label, "SCOUT_" + label]
    )

    return module_features, module_labels


def module_connectivity_fc(rois, label, fc):
    int_rois = np.array(rois)
    ext_rois = np.setdiff1d(np.array([i for i in range(len(fc))]), int_rois)

    fc_int = np.nanmean(fc[int_rois, :][:, int_rois], dtype=float)
    fc_out = np.nanmean(fc[int_rois[:, None], ext_rois], dtype=float)
    module_features = np.array([fc_int, fc_out])
    module_labels = np.array(["FCINT_" + label, "FCOUT_" + label])

    return module_features, module_labels


def tree_connectivity_fc(tree_dictionary, fc):
    t_features = np.array([])
    t_features_names = np.array([])
    for mod in tree_dictionary:
        rois_in_clust = tree_dictionary[mod]
        if len(rois_in_clust) > 1:
            l_features, l_names = module_connectivity_fc(rois_in_clust, mod, fc)
            t_features = np.hstack([t_features, l_features])
            t_features_names = np.hstack([t_features_names, l_names])
    return t_features, t_features_names


def tree_connectivity(tree_dictionary, sc, fc):
    t_features = np.array([])
    t_features_names = np.array([])
    for mod in tree_dictionary:
        rois_in_clust = tree_dictionary[mod]
        if len(rois_in_clust) > 1:
            l_features, l_names = module_connectivity(rois_in_clust, mod, sc, fc)
            t_features = np.hstack([t_features, l_features])
            t_features_names = np.hstack([t_features_names, l_names])
    return t_features, t_features_names


def get_module_vol(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_vol = np.where(atlas_data == (np.array(rois) + 1), value, 0).sum(axis=3)
    return module_vol


def add_gamma_to_lvl_dict(dict, g):
    oldkeys = list(dict.keys())
    newkeys = [
        s.replace("lvl_", "gamma_" + str(round(g, 2)) + "_lvl_") for s in oldkeys
    ]
    vals = list(dict.values())
    newdictionary = {k: v for k, v in zip(newkeys, vals)}
    return newdictionary