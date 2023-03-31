"""
Functions for building dendrogram trees and manage their levels
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import warnings


def tree_modules(W, num_clust):
    Z = linkage(W, "average")
    T = fcluster(Z, num_clust, criterion="maxclust")
    return T


def level_dictionary(T, lvl):
    l_dict = {}
    for i in range(1, lvl + 1):
        rois_in_clust = np.where(T == i)[0]
        if len(rois_in_clust) != 0:
            desc = "lvl_" + str(lvl) + "_mod_" + str(i)
            l_dict[desc] = rois_in_clust.tolist()
        else:
            warnings.warn(
                "Empty cluster found in level " + str(lvl) + ", module " + str(i) + "!"
            )
    return l_dict


def tree_dictionary(init_level, end_level, W, tree_class="reduced"):
    t_dict = {}
    for i in range(init_level, end_level + 1):
        T = tree_modules(W, i)
        level_dict = level_dictionary(T, i)
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


def level_from_tree(tree, level_num):
    level_keys = list(
        filter(lambda x: x.startswith("lvl_" + str(level_num) + "_"), tree.keys())
    )
    level = [tree[key] for key in level_keys]
    labels = [key for key in level_keys]
    return level, labels


def T_from_level(level):
    T = np.zeros(max(map(max, level)) + 1)
    for idx, mod in enumerate(level):
        T[mod] = idx + 1
    return T
