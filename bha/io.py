import numpy as np
import scipy.sparse
import glob


def load_data(path):
    files = glob.glob(path)
    matrices = np.array(
        [scipy.sparse.load_npz(f).toarray(dtype=np.float16) for f in sorted(files)]
    )
    return matrices
