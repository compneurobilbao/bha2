from scipy.spatial.distance import pdist
import numpy as np

def connectome_average(fc_all, sc_all):
    fcm = np.median(fc_all, axis=0)
    scm = np.median(sc_all, axis=0)
    return fcm, scm


def matrix_fusion(g, fcm, scm):
    if g == 0.0:
        cc = scm
    elif g == 1.0:
        cc = fcm
    else:
        cc = (g * abs(fcm)) + ((1 - g) * scm)

    cc_dist = pdist(cc, "cosine")
    W = cc_dist / max(cc_dist)
    return W


def density_threshold(W, density):
    W_thr = np.zeros(W.shape)
    W_sorted = np.sort(abs(W.flatten()))
    W_thr[np.where(abs(W) > W_sorted[int((1 - density) * len(W_sorted))])] = 1
    return W * W_thr


def get_module_matrix(matrix, rois):
    module_matrix = matrix[rois, :][:, rois]
    return module_matrix


def get_module_matrix_external(matrix, rois):
    ext_rois = np.setdiff1d(np.array([i for i in range(len(matrix))]), rois)
    module_matrix = matrix[rois][:, ext_rois]
    return module_matrix


def similarity_mean_level(fcm, scm, level):

    similarities = []
    for rois in level:
        if len(rois) > 1:
            mod_fc = get_module_matrix(fcm, rois)
            mod_sc = get_module_matrix(scm, rois)
            mod_fc_bin = np.where(abs(mod_fc) > 0, 1, 0)
            mod_sc_bin = np.where(mod_sc > 0, 1, 0)

            if (mod_fc_bin.sum() + mod_sc_bin.sum()) != 0:
                sim = (
                    2
                    * np.multiply(mod_fc_bin, mod_sc_bin).sum()
                    / (mod_fc_bin.sum() + mod_sc_bin.sum())
                )
                similarities.append(sim)
            else:
                similarities.append(np.nan)
    return np.nanmean(similarities)


def modularity(A, T):
    N = len(T)
    K = np.array(A.sum(axis=0).reshape(1, -1), dtype=np.float64)
    m = K.sum()
    B = A - (K.T * K) / m
    s = np.array([T for i in range(N)], dtype=np.float64)
    Q = B[np.where((s.T - s) == 0)].sum() / m
    return Q
