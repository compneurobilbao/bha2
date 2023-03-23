import numpy as np
import nibabel as nib
import pandas as pd

def get_atlas_rois_from_mask(mask, atlas):
    mask_data = mask.get_fdata()
    atlas_data = atlas.get_fdata()
    mask_rois = np.setdiff1d(np.unique(atlas_data[mask_data == 1]), 0)
    return mask_rois


def get_module_vol(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_vol = np.where(atlas_data == (np.array(rois) + 1), value, 0).sum(axis=3)
    return module_vol

def get_atlas_coords(atlas_path): 
    atlas = nib.load(atlas_path)
    atlas_voxels = (
        nib
        .affines
        .apply_affine(atlas.affine, np.transpose(atlas.get_fdata().nonzero()))
        .astype(int)
    )
    atlas_roivals = atlas.get_fdata()[atlas.get_fdata() != 0].astype(int)
    atlas_coords = pd.DataFrame(atlas_voxels, columns=["x", "y", "z"], index=atlas_roivals)
    atlas_centroids = atlas_coords.groupby(atlas_coords.index).mean().astype(int)
    return atlas_centroids

def image_overlaps(vol1, vol2):
    intersect = np.where((vol1 != 0) & (vol2 != 0), 1, 0)
    overlap = intersect.sum() / vol1.sum()
    return overlap