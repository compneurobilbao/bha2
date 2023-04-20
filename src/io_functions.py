import pandas as pd
import numpy as np
import glob


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
