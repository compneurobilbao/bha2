import pandas as pd
import numpy as np
import glob
from neuromaps.parcellate import Parcellater

def load_data(path):
    files = glob.glob(path + "/*.csv")
    matrices = np.array(
        [
            pd.read_csv(
                f, header=None, delimiter=" ", dtype=np.float16, engine="c"
            ).to_numpy(dtype=np.float16)
            for f in sorted(files)
        ]
    )
    return matrices

def load_receptor_data(path, partition):
    files = glob.glob(path + "/*.nii.gz")
    parcellater = Parcellater(partition, 'MNI152')
    parcellated = {}
    for receptor in sorted(files):
        name = receptor.split('/')[-1]  # get nifti file name
        name = name.split('.')[0]  # remove .nii
        parcellated[name] = parcellater.fit_transform(receptor, 'MNI152', True)[0]
    return parcellated