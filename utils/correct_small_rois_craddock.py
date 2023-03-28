import sys
sys.path.append('/workspaces/bha2/src')

import numpy as np
import nibabel as nib
import glob 
import os
from scipy.ndimage import binary_dilation
from ni_functions import get_atlas_coords, image_overlaps

def select_rois_from_lobe_partitions(lp_path, area):
    image_names = glob.glob(os.path.join(lp_path, area, '*.nii.gz'))

    image_roi_labels = []
    image_roi_volumes = []
    image_mean_roi_vol = []

    for im_name in image_names:
        image = nib.load(im_name)
        image_data = image.get_fdata()
        rois = np.setdiff1d(np.unique(image_data), 0).astype(int)

        roi_vol = np.zeros((len(rois)))
        for idx, roi_number in enumerate(rois):
            roi_vol[idx] = np.sum(image_data == roi_number)
        image_roi_volumes.append(roi_vol)
        image_mean_roi_vol.append(np.mean(roi_vol))
        image_roi_labels.append(rois)
    return image_names, image_roi_labels, image_roi_volumes, image_mean_roi_vol

def small_roi_correction(lp_path, area, vol_des, min_vol):
    image_names, image_roi_labels, image_roi_volumes, image_mean_roi_vol = select_rois_from_lobe_partitions(lp_path, area)
    size_near_desired = np.argmin(
        np.abs(np.array(image_mean_roi_vol) - vol_des))
    image_roi_vol_selected = image_roi_volumes[size_near_desired]
    image_selected_orig = nib.load(image_names[size_near_desired])
    small_rois = image_roi_labels[size_near_desired][np.where(
        image_roi_vol_selected < min_vol)[0]]

    vol_to_correct = image_selected_orig.get_fdata()
    if len(small_rois) > 0:
        for sr in small_rois:
            sr_loc = np.where(vol_to_correct ==
                              sr, 1, 0).astype(bool)
            sr_dil_loc = binary_dilation(sr_loc, iterations=1)
            neighbours = np.setdiff1d(
                np.unique(vol_to_correct[sr_dil_loc]), [0, sr]).astype(int)
            if len(neighbours) == 0:
                sr_dil_loc = binary_dilation(sr_loc, iterations=2)
                neighbours = np.setdiff1d(
                    np.unique(vol_to_correct[sr_dil_loc]), [0, sr]).astype(int)
            if len(neighbours) > 0:
                size_neighbours = image_roi_vol_selected[np.where(
                    np.in1d(image_roi_labels[size_near_desired], neighbours))[0]]
                small_neighbour_idx = np.argmin(size_neighbours)
                vol_to_correct[sr_loc] = neighbours[small_neighbour_idx]
            else:
                vol_to_correct[sr_loc] = 0
    return vol_to_correct, image_selected_orig

def get_roi_anatomical_description(atlas):
    coords = get_atlas_coords(atlas)
    atlas_data = atlas.get_fdata()
    atlas_rois = np.setdiff1d(np.unique(atlas_data), 0).astype(int)

    idx_area = 0
    for area in brain_areas:
        for roi_label in atlas_rois:
            area_mask_L = nib.load(os.path.join('/workspaces', 'bha2', 'brain_templates',
                                'lobule_masks', 'masks_splitted_in_hemispheres', area + '_L_mask.nii.gz'))
            area_mask_L_data = area_mask_L.get_fdata()
            area_mask_R = nib.load(os.path.join('/workspaces', 'bha2', 'brain_templates',
                                'lobule_masks', 'masks_splitted_in_hemispheres', area + '_R_mask.nii.gz'))
            area_mask_R_data = area_mask_R.get_fdata()
            roi_mask = np.where(atlas_data == roi_label, 1, 0)
            coords.loc[roi_label, area + '_L'] = image_overlaps(roi_mask, area_mask_L_data)
            coords.loc[roi_label, area + '_R'] = image_overlaps(roi_mask, area_mask_R_data)

        idx_area += 2

    coords.index.name = 'ROI_number'
    return coords


###############################################################################################
# Main
###############################################################################################

volume_per_roi_desired = int(sys.argv[1])
min_volume_per_roi = int(sys.argv[2])


LOBE_PARTITION_PATH = "/workspaces/bha2/data/craddock/all_lobe_partitions"
brain_areas = ['Frontal', 'Parietal', 'Occipital', 'Temporal', 'Insula',  'Subcortical']
template_empty = np.zeros(nib.load(os.path.join('/workspaces', 'bha2', 
                                                'brain_templates', 'lobule_masks', 'Frontal_mask.nii.gz')).shape)

FINAL_ROI_LABEL_COUNTER = 1
for b_area in brain_areas:

    vol_for_small_rois_correction, image_selected  = small_roi_correction(LOBE_PARTITION_PATH, 
                                                                          b_area, volume_per_roi_desired, min_volume_per_roi)
    new_roi_labels = np.setdiff1d(
        np.unique(vol_for_small_rois_correction), 0).astype(int)
    for roi in new_roi_labels:
        template_empty[vol_for_small_rois_correction ==
                       roi] = FINAL_ROI_LABEL_COUNTER
        FINAL_ROI_LABEL_COUNTER += 1

nib.save(nib.Nifti1Image(template_empty, image_selected.affine), os.path.join(
    '/workspaces', 'bha2', 'brain_templates', 'craddock_' + str(FINAL_ROI_LABEL_COUNTER-1) + '.nii.gz'))

atlas_coords = get_roi_anatomical_description(nib.Nifti1Image(template_empty, image_selected.affine))

atlas_coords.to_csv(os.path.join('/workspaces', 'bha2', 'brain_templates', 'craddock_' + str(FINAL_ROI_LABEL_COUNTER-1) + '_rois.csv'))