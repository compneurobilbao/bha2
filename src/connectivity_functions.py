from scipy.spatial.distance import pdist, squareform
from src.tree_functions import level_from_tree, T_from_level
import numpy as np



def connectome_average(fc_all, sc_all):
    fcm = np.median(fc_all, axis=0)
    scm = np.median(sc_all, axis=0)
    return fcm, scm

def density_threshold(W, density):
    W_thr = np.zeros(W.shape)
    W_sorted = np.sort(abs(W.flatten()))
    W_thr[np.where(abs(W) > W_sorted[int((1 - density) * len(W_sorted))])] = 1
    return W * W_thr

def remove_rois_from_connectomes(rois, fcm, scm):
    fcm_rois_rem = np.delete(fcm, rois, axis=0)
    fcm_rois_rem = np.delete(fcm_rois_rem, rois, axis=1)
    scm_rois_rem = np.delete(scm, rois, axis=0)
    scm_rois_rem = np.delete(scm_rois_rem, rois, axis=1)
    return fcm_rois_rem, scm_rois_rem

def equal_clean_connectomes(fcm, scm):
    zero_rows_sc = np.where(~scm.any(axis=1))[0]
    fcm_nonzero, scm_nonzero = remove_rois_from_connectomes(zero_rows_sc, fcm, scm)
    scm_density = np.where(scm_nonzero.flatten() > 0, 1, 0).sum(dtype=float) / (
        len(scm_nonzero.flatten())
    )
    fcm_thr = density_threshold(fcm_nonzero, scm_density)
    zero_rows_fc = np.where(~fcm_thr.any(axis=1))[0]
    fcm_nonzero, scm_nonzero = remove_rois_from_connectomes(
        zero_rows_fc, fcm_thr, scm_nonzero
    )
    return fcm_nonzero, scm_nonzero, zero_rows_fc, zero_rows_sc

def matrix_fusion(g, fcb, scb):
    if g == 0.0:
        cc = scb
    elif g == 1.0:
        cc = fcb
    else:
        cc = (g * fcb) + ((1 - g) * scb)
    return cc

def get_module_matrix(matrix, rois):
    module_matrix = matrix[rois, :][:, rois]
    return module_matrix


def get_module_matrix_external(matrix, rois):
    ext_rois = np.setdiff1d(np.array([i for i in range(len(matrix))]), rois)
    module_matrix = matrix[rois][:, ext_rois]
    return module_matrix


def similarity_level(fcm, scm, level):
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
    return similarities


def modularity(A, T):
    N = len(T)
    K = np.array(A.sum(axis=0).reshape(1, -1), dtype=np.float64)
    m = K.sum()
    B = A - (K.T * K) / m
    s = np.array([T for i in range(N)], dtype=np.float64)
    Q = B[np.where((s.T - s) == 0)].sum() / m
    return Q

def x_modularity(tree, l, fcm, scm):
    level, labels = level_from_tree(tree, l)
    T = T_from_level(level)
    sim = np.nanmean(similarity_level(fcm, scm, level))
    mod_sc = modularity(scm, T)
    mod_fc = modularity(fcm, T)
    x = pow((sim * mod_sc * mod_fc), (1 / 3))
    return x

def local_modularity(A, T):
    m = sum(A.flatten())
    ext_T = np.setdiff1d(np.array([i for i in range(len(A))]), T)
    mc = A[T][:, T]
    mc_m = sum(mc.flatten()) / m
    ec = A[T][:, ext_T]
    Q_m = mc_m - np.power((2 * sum(mc.flatten()) + sum(ec.flatten())) / (2 * m), 2)
    return Q_m
