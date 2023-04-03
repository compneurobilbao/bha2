import sys
import os
from make_local_connectivity_scorr import *
from binfile_parcellation import *
from make_image_from_bin import *
from make_image_from_bin_renum import *
from time import time

path_fmriproc = sys.argv[1]
path_outmaps = sys.argv[2]
subject = sys.argv[3]
mask = sys.argv[4]

# the name of the maskfile that we will be using
maskname = os.path.join(path_outmaps,mask,mask+'_mask.nii.gz')
# make a list of all of the input fMRI files that we will be using
in_file = os.path.join(path_fmriproc,subject,subject+'_preprocessed.nii.gz')


# construct the connectivity matrices using scorr and a r>0.5 threshold
scorr_mat = os.path.join(path_outmaps,mask,subject,'rm_scorr_conn.npy')

# call the funtion to make connectivity
make_local_connectivity_scorr( in_file, maskname, scorr_mat, 0.5 )
