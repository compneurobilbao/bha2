"""Conde for building dendrogram trees for a given neuroimaging dataset which includes
Structural Connectivity (SC) and Functional Connectivity (FC) matrices."""

from src.tree_functions import *
from src.connectivity_functions import *
from src.io_functions import load_data
import os
import sys
import json
import nibabel as nib

# Input variables
project_path = sys.argv[1]
conn_size = int(sys.argv[2])
tree_lower = int(sys.argv[3])
tree_upper = int(sys.argv[4])
tree_class = sys.argv[5]

# Checking if fcm and scm are stored in tmp folder
if os.path.exists(os.path.join(project_path, "tmp", "n" + str(conn_size) + "_fcm.npy")):
    fcm = np.load(os.path.join(project_path, "tmp", "n" + str(conn_size) + "_fcm.npy"))
    scm = np.load(os.path.join(project_path, "tmp", "n" + str(conn_size) + "_scm.npy"))
    print("fcm and scm loaded from tmp folder")
else:
    sc_group = load_data(
        os.path.join(project_path, "data", "iPA_" + str(conn_size), "sc")
    )
    fc_group = load_data(
        os.path.join(project_path, "data", "iPA_" + str(conn_size), "fc")
    )
    fcm, scm = connectome_average(fc_group, sc_group)
    if not os.path.exists(os.path.join(project_path, "tmp")):
        os.mkdir(os.path.join(project_path, "tmp"))
    np.save(os.path.join(project_path, "tmp", "n" + str(conn_size) + "_fcm.npy"), fcm)
    np.save(os.path.join(project_path, "tmp", "n" + str(conn_size) + "_scm.npy"), scm)

# Equalizing both connectome densities and removing the nodes disconnected from the network
fcm_clean, scm_clean, fc_removed_rois, sc_removed_rois = equal_clean_connectomes(
    fcm, scm
)

# Binzarizing both connectomes for building the tree
fcm_bin = np.where(abs(fcm_clean) > 0, 1, 0)
scm_bin = np.where(scm_clean > 0, 1, 0)


# Loading the original parcellation
parcellation = nib.load(
    os.path.join(
        project_path,
        "data",
        "iPA_" + str(conn_size),
        "iPA_" + str(conn_size) + ".nii.gz",
    )
)
parcellation_vol = parcellation.get_fdata()

# Defining an empty matrix to store the parcellation without the rows removed
parcellation_clean = np.zeros(
    (parcellation_vol.shape[0], parcellation_vol.shape[1], parcellation_vol.shape[2])
)

# Removing the not connected ROIs in the original parcellation, it is important to remove first the SC ROIs
old_rois = np.arange(1, parcellation_vol.max() + 1, dtype=int)
old_rois = np.delete(old_rois, sc_removed_rois)
old_rois = np.delete(old_rois, fc_removed_rois)

# Assigning the new ROI numbers to the parcellation
for idx, rois in enumerate(old_rois):
    parcellation_clean[parcellation_vol == rois] = idx + 1

parcellation_clean_img = nib.Nifti1Image(parcellation_clean, affine=parcellation.affine)

if not (
    os.path.exists(
        os.path.join(project_path, "data", "iPA_" + str(conn_size), "processed")
    )
):
    os.mkdir(os.path.join(project_path, "data", "iPA_" + str(conn_size), "processed"))

nib.save(
    parcellation_clean_img,
    os.path.join(
        project_path,
        "data",
        "iPA_" + str(conn_size),
        "processed",
        "initial_parcellation.nii.gz",
    ),
)

# Building the tree
for g in np.arange(0, 1.1, 0.1):
    W = matrix_fusion(g, fcm_bin, scm_bin)
    t_dict = tree_dictionary(tree_lower, tree_upper, W, tree_class)
    json.dump(
        t_dict,
        open(
            os.path.join(
                project_path,
                "data",
                "iPA_" + str(conn_size),
                "processed",
                "tree_" + tree_class + "_g_" + str(round(g, 2)) + ".json",
            ),
            "w",
        ),
    )
