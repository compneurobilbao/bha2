import numpy as np
import pandas as pd
import nibabel as nib
import os
import sys

volume_per_roi_desired = int(sys.argv[1])
participants = pd.read_csv(
    os.path.join("/workspaces", "bha2", "data", "participants.tsv"), sep="\t"
)
brain_lobules = [
    "Frontal",
    "Parietal",
    "Occipital",
    "Temporal",
    "Insula",
    "Subcortical",
]
path_to_rest_prep = os.path.join(
    "/workspaces", "bha2", "data", "processed", "rest_prep"
)
path_to_parcellations = os.path.join(
    "/workspaces", "bha2", "data", "processed", "pyClusterROI_parcellations"
)
template_proportion = np.zeros(
    nib.load(
        os.path.join(
            "/workspaces",
            "bha2",
            "brain_templates",
            "lobule_masks",
            "Frontal_mask.nii.gz",
        )
    ).shape
)

for sub in participants.values[:, 0]:
    rest_prep = nib.load(
        os.path.join(path_to_rest_prep, sub, sub + "_preprocessed.nii.gz")
    ).get_fdata()
    rest_prep_avg_mask = np.where(np.mean(rest_prep, axis=3) != 0, 1, 0)
    template_proportion += rest_prep_avg_mask

population_rsfmri_mask = np.where(
    template_proportion / len(participants.values[:, 0]) > 0.5, 1, 0
)

for lob in brain_lobules:
    if not os.path.exists(path_to_parcellations):
        os.mkdir(path_to_parcellations)
    if not os.path.exists(os.path.join(path_to_parcellations, lob)):
        os.mkdir(os.path.join(path_to_parcellations, lob))
    lob_mask = nib.load(
        os.path.join(
            "/workspaces",
            "bha2",
            "brain_templates",
            "lobule_masks",
            lob + "_mask.nii.gz",
        )
    )
    lob_mask_crop = nib.Nifti1Image(
        lob_mask.get_fdata() * population_rsfmri_mask, lob_mask.affine, lob_mask.header
    )
    nib.save(
        lob_mask_crop, os.path.join(path_to_parcellations, lob, lob + "_mask.nii.gz")
    )
    n_clusters = np.round(
        np.mean(lob_mask_crop.get_fdata().flatten())
        * np.prod(lob_mask_crop.shape)
        / volume_per_roi_desired
    ).astype(int)

    for sub in participants.values[:, 0]:
        if not os.path.exists(os.path.join(path_to_parcellations, lob, sub)):
            os.mkdir(os.path.join(path_to_parcellations, lob, sub))
        os.system(
            "python2.7 /workspaces/bha2/utils/pyClusterROI/pyClusterROI_individual.py "
            + path_to_rest_prep
            + " "
            + path_to_parcellations
            + " "
            + sub
            + " "
            + lob
        )

    os.system(
        "python2.7 /workspaces/bha2/utils/pyClusterROI/pyClusterROI_group_and_convertNII.py "
        + path_to_parcellations
        + " "
        + os.path.join("/workspaces", "bha2", "data", "participants.tsv")
        + " "
        + lob
        + " "
        + str(n_clusters)
    )
