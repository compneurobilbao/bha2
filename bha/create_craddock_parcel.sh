#!/bin/bash


source activate neuro

(ls -lrta /project/ | grep '^d' | grep 'sub' | awk '{ print $9 }') > /project/participants_list.txt

fslmaths /app/brain_templates/lobule_masks/Frontal_mask.nii.gz -sub \
    /app/brain_templates/lobule_masks/Frontal_mask.nii.gz /project/population_rsfmri_proportion.nii.gz

for p in $(cat /project/participants_list.txt)
do
    fslmaths /project/${p}/${p}_preprocessed.nii.gz -Tstd -bin /project/${p}/mask
    fslmaths /project/population_rsfmri_proportion.nii.gz -add  /project/${p}/mask /project/population_rsfmri_proportion.nii.gz
done

n_s=$(wc -l /project/participants_list.txt | awk '{ print $1 }')

fslmaths /project/population_rsfmri_proportion.nii.gz -div $n_s -thr 0.5 -bin \
    /project/population_rsfmri_mask.nii.gz

for lob in Frontal Parietal Occipital Temporal Insula Subcortical Cerebellum Brain_stem
do
    mkdir -p /project/Craddock_partition/${lob}
    
    fslmaths /app/brain_templates/lobule_masks/${lob}_mask.nii.gz -mas \
        /project/population_rsfmri_mask.nii.gz /project/Craddock_partition/${lob}/${lob}_mask.nii.gz
    dim1_img=$(fslval /project/Craddock_partition/${lob}/${lob}_mask.nii.gz dim1)
    dim2_img=$(fslval /project/Craddock_partition/${lob}/${lob}_mask.nii.gz dim2)
    dim3_img=$(fslval /project/Craddock_partition/${lob}/${lob}_mask.nii.gz dim3)
    mean_img=$(fslstats /project/Craddock_partition/${lob}/${lob}_mask.nii.gz -m)
    #dim_caler is a factor to scale the dimension of the mask to get the aproximated desired
    #size of the parcels. Here as we used 2x2x2mm voxels, and we want to have parcels of 75 voxels,
    #dim_scaler is 
    dim_scaler=75
    Nclust=$(echo "${mean_img} * ${dim1_img} * ${dim2_img} * ${dim3_img} / ${dim_scaler}" | bc -l) 
    
    for p in $(cat /project/participants_list.txt)
    do
        mkdir -p /project/Craddock_partition/${lob}/${p}
        python /app/utils/craddockParcel/pyClusterROI_individual.py $p $lob
    done

    python /app/utils/craddockParcel/pyClusterROI_group_and_convertNII.py /project/participants_list.txt $lob $Nclust
done