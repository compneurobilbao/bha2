import numpy as np
import nibabel as nib
import pandas as pd


def get_atlas_rois_from_mask(mask, atlas):
    mask_data = mask.get_fdata()
    atlas_data = atlas.get_fdata()
    mask_rois = (np.setdiff1d(np.unique(atlas_data[mask_data == 1]), 0)).astype(int)
    return mask_rois


def get_module_vol(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_vol = np.isin(atlas_data, (np.array(rois) + 1)) * value
    return module_vol


def get_atlas_coords(atlas):
    atlas_voxels = nib.affines.apply_affine(
        atlas.affine, np.transpose(atlas.get_fdata().nonzero())
    ).astype(int)
    atlas_roivals = atlas.get_fdata()[atlas.get_fdata() != 0].astype(int)
    atlas_coords = pd.DataFrame(
        atlas_voxels, columns=["x", "y", "z"], index=atlas_roivals
    )
    atlas_centroids = atlas_coords.groupby(atlas_coords.index).mean().astype(int)
    return atlas_centroids


def image_overlaps(vol1, vol2):
    intersect = np.where((vol1 != 0) & (vol2 != 0), 1, 0)
    overlap = intersect.sum() / vol1.sum()
    return overlap


def distance_between_modules(module_A, module_B, atlas):
    atlas_coords = get_atlas_coords(atlas)
    d_list = []
    for roi_A in module_A:
        roi_A_coords = atlas_coords.loc[roi_A]
        for roi_B in module_B:
            roi_B_coords = atlas_coords.loc[roi_B]
            d_list.append(np.linalg.norm(roi_A_coords - roi_B_coords))

    return np.median(d_list)
