import sys
import os
from make_local_connectivity_scorr import *
from binfile_parcellation import *
from group_mean_binfile_parcellation import *
from group_binfile_parcellation import *
from make_image_from_bin import *
from make_image_from_bin_renum import *
from time import time

path_fmriproc = "/project"
path_outmaps = "/project/Craddock_partition"

subject_list = sys.argv[1]
mask = sys.argv[2]
Nclust = int(round(float(sys.argv[3])))

# the name of the maskfile that we will be using
maskname = os.path.join(path_outmaps,mask,mask+'_mask.nii.gz')

# for both group-mean and 2-level clustering we need to know the number of
# nonzero voxels in in the mask
NUM_CLUSTERS = [int(round(Nclust*0.90)), Nclust, int(round(Nclust*1.10))]
mask_voxels=(nb.load(maskname).get_data().flatten()>0).sum()

# now group mean cluster scorr files
subjects_file = open(subject_list)
scorr_conn_files = []
for line in subjects_file:
    scorr_conn_files.append(os.path.join(path_outmaps, mask, line.replace('\n',''),  'rm_scorr_conn.npy'))
subjects_file.close()

group_mean_scorr = os.path.join(path_outmaps, mask, 'rm_group_mean_scorr_cluster')
group_mean_binfile_parcellate(scorr_conn_files, group_mean_scorr, NUM_CLUSTERS, mask_voxels)


for k in NUM_CLUSTERS:
    binfile = os.path.join(path_outmaps, mask, 'rm_group_mean_scorr_cluster_'+str(k)+'.npy')
    imgfile = os.path.join(path_outmaps, mask, mask+'_group_mean_'+str(k)+'.nii.gz')
    make_image_from_bin_renum(imgfile,binfile,maskname)

