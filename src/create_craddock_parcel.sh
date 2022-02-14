#!/bin/bash


source activate neuro

(ls -lrta /project/ | grep '^d' | grep 'sub' | awk '{ print $9 }') > /project/participants_list.txt


for lob in Frontal Parietal Occipital Temporal Insula Subcortical Cerebellum Brain_stem
do
    dim1_img=$(fslval /app/brain_templates/lobule_masks/${lob}_mask.nii.gz dim1)
    dim2_img=$(fslval /app/brain_templates/lobule_masks/${lob}_mask.nii.gz dim1)
    dim3_img=$(fslval /app/brain_templates/lobule_masks/${lob}_mask.nii.gz dim1)
    mean_img=$(fslstats /app/brain_templates/lobule_masks/${lob}_mask.nii.gz -m)
    Nclust=$(echo "${mean_img} * ${dim1_img} * ${dim2_img} * ${dim3_img} / 20" | bc -l) 
    
    for p in $(cat /project/participants_list.txt)
    do
        python /app/utils/pyClusterROI_individual.py $p $lob
    done

    python /app/utils/pyClusterROI_group_and_convertNII.py /project/participants_list.txt $lob $Nclust
done