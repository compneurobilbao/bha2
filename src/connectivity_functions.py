"""Definintion of functions related to connectomes and connectivity measures."""

from scipy.spatial.distance import pdist, squareform
from src.tree_functions import level_from_tree, T_from_level
import numpy as np


def connectome_average(fc_all, sc_all):
    """
    Compute the average connectivity matrices for a set of subjects.

    Parameters
    ----------
    fc_all : ndarray, shape (N, M, M)
        A set of functional connectivity matrices for N subjects.

    sc_all : ndarray, shape (N, M, M)
        A set of structural connectivity matrices for N subjects.

    Returns
    -------
    fcm : ndarray, shape (M, M)
        The average functional connectivity matrix.

    scm : ndarray, shape (M, M)
        The average structural connectivity matrix.
    """
    fcm = np.median(fc_all, axis=0)
    scm = np.median(sc_all, axis=0)
    return fcm, scm


def density_threshold(W, density):
    """
    Apply density thresholding to a given matrix.

    Parameters:
    ----------
    W : ndarray, shape (M, M)
        Matrix to threshold.
    density : float
        Density threshold value between 0 and 1.

    Returns:
    -------
    ndarray
        Thresholded matrix.
    """
    W_thr = np.zeros(W.shape)
    W_sorted = np.sort(abs(W.flatten()))
    W_thr[np.where(abs(W) > W_sorted[int((1 - density) * len(W_sorted))])] = 1
    return W * W_thr


def remove_rois_from_connectomes(rois, fcm, scm):
    """
    Remove rows and columns from a functional and structural connectome.

    Parameters
    ----------
    rois : list
        List of ROIs to remove from the connectomes.
    fcm : ndarray, shape (M, M)
        Functional connectivity matrix.
    scm : ndarray, shape (M, M)
        Structural connectivity matrix.

    Returns
    -------
    fcm_rois_rem : ndarray, shape (M-rois, M-rois)
        Functional connectivity matrix with ROIs removed.
    scm_rois_rem : ndarray, shape (M-rois, M-rois)
        Structural connectivity matrix with ROIs removed.
    """
    fcm_rois_rem = np.delete(fcm, rois, axis=0)
    fcm_rois_rem = np.delete(fcm_rois_rem, rois, axis=1)
    scm_rois_rem = np.delete(scm, rois, axis=0)
    scm_rois_rem = np.delete(scm_rois_rem, rois, axis=1)
    return fcm_rois_rem, scm_rois_rem


def equal_clean_connectomes(fcm, scm):
    """
    Remove zero rows from SCM and FCM, and threshold FCM to match density of SCM
    
    Parameters
    ----------
    fcm : ndarray
        Functional connectivity matrix
    scm : ndarray
        Structural connectivity matrix

    Returns
    -------
    fcm_nonzero: ndarray
        Functional connectivity matrix with zero rows removed.
    scm_nonzero: ndarray
        Structural connectivity matrix with zero rows removed.
    zero_rows_fc: ndarray
        The indices of the zero rows in the original fcm.
    zero_rows_sc: ndarray
        The indices of the zero rows in the original scm.
    """
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
    """
    Perform matrix fusion based on the given parameters.

    Parameters
    ----------
    g : float
        The fusion parameter, ranging from 0.0 to 1.0.
    fcb : ndarray, shape (M,M)
        The first connectivity matrix.
    scb : ndarray, shape (M,M)
        The second connectivity matrix.

    Returns
    -------
    cc : ndarray
        The fused connectivity matrix.

    """
    if g == 0.0:
        cc = scb
    elif g == 1.0:
        cc = fcb
    else:
        cc = (g * fcb) + ((1 - g) * scb)
    return cc


def get_module_matrix(matrix, rois):
    """
    Returns a module matrix based on the given matrix and regions of interest (ROIs).

    Parameters:
    ----------
    matrix : ndarray
        The input matrix.
    rois : list
        The list of regions of interest.

    Returns:
    -------
    module_matrix: ndarray
        The module matrix.

    """
    module_matrix = matrix[rois, :][:, rois]
    return module_matrix


def get_module_matrix_external(matrix, rois):
    """
    Get the module matrix for external regions (regions not included in rois list).

    Parameters:
    ----------
    matrix : ndarray
        The input matrix.
    rois : list
        The list of regions of interest.

    Returns:
    -------
    module_matrix : ndarray
        The external connectivity of the module.

    """
    ext_rois = np.setdiff1d(np.array([i for i in range(len(matrix))]), rois)
    module_matrix = matrix[rois][:, ext_rois]
    return module_matrix


def similarity_level(fcm, scm, level):
    """
    Calculate the similarity level between two connectivity matrices for a given level.

    Parameters:
    ----------
    fcm : ndarray, shape (M, M)
        Functional connectivity matrix.
    scm : ndarray, shape (M, M)
        Structural connectivity matrix.
    level : dictionary
        A dictionary containing the ROIs included in every module for a given level.

    Returns:
    -------
    similarities: list
        A list of similarity values for each ROI in the given level.
    """
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
    """
    Calculate the Newmann modularity of a network given the adjacency matrix 
    A and the community assignment vector T.

    Parameters:
    A : ndarray, shape(M,M) 
        Adjacency matrix of the network.
    T : ndarray, shape(1,M) 
        Community assignment vector.

    Returns:
    Q : float
        Modularity value of the network.
    """
    N = len(T)
    K = np.array(A.sum(axis=0).reshape(1, -1), dtype=np.float64)
    m = K.sum()
    B = A - (K.T * K) / m
    s = np.array([T for i in range(N)], dtype=np.float64)
    Q = B[np.where((s.T - s) == 0)].sum() / m
    return Q


def x_modularity(tree, l, fcm, scm):
    """
    Calculate the cross-modularity between structure and function of a level of the tree

    Parameters:
    ----------
    tree : dictionary
        The tree dictionary representing the hierarchical clustering.
    l : int
        The level at which to calculate the x-modularity.
    fcm : ndarray, shape (M, M)
        Functional connectivity matrix.
    scm : ndarray, shape (M, M)
        Structural connectivity matrix.

    Returns:
    -------
    x: float
        The cross-modularity value.

    """
    level, labels = level_from_tree(tree, l)
    T = T_from_level(level)
    sim = np.nanmean(similarity_level(fcm, scm, level))
    mod_sc = modularity(scm, T)
    mod_fc = modularity(fcm, T)
    x = pow((sim * mod_sc * mod_fc), (1 / 3))
    return x