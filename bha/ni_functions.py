import numpy as np


def get_atlas_rois_from_mask(mask, atlas):
    mask_data = mask.get_fdata()
    atlas_data = atlas.get_fdata()
    mask_rois = np.setdiff1d(np.unique(atlas_data[mask_data == 1]), 0)
    return mask_rois


def get_module_vol(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_vol = np.where(atlas_data == (np.array(rois) + 1), value, 0).sum(axis=3)
    return module_vol
