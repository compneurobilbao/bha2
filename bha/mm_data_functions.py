import numpy as np
import pandas as pd
from scipy.stats import zscore


def generate_receptor_matrix(receptor_dict):
    receptor_names = np.array(
        [
            "5HT1a",
            "5HT1b",
            "5HT2a",
            "5HT4",
            "5HT6",
            "5HTT",
            "A4B2",
            "CB1",
            "D1",
            "D2",
            "DAT",
            "GABAa",
            "H3",
            "M1",
            "mGluR5",
            "MU",
            "NAT",
            "NMDA",
            "VAChT",
        ]
    )
    nnodes = len(receptor_dict[list(receptor_dict.keys())[0]])
    receptor_mat = np.zeros([nnodes, len(receptor_names)])
    image_names = list(receptor_dict.keys())
    for i, r in enumerate(receptor_names):
        images_of_receptor = list(filter(lambda x: x.startswith(r), image_names))
        if len(images_of_receptor) == 1:
            receptor_mat[:, i] = receptor_dict[images_of_receptor[0]]
        elif len(images_of_receptor) > 1:
            scaler = 0
            for im in images_of_receptor:
                weight = float((im.split("hc")[-1]).split("_")[0])
                scaler += weight
                receptor_mat[:, i] += zscore(receptor_dict[im]) * weight
            receptor_mat[:, i] /= scaler
        else:
            raise ValueError("Receptor not found")
    receptor_df = pd.DataFrame(
        receptor_mat,
        columns=receptor_names,
        index=np.array(["module" + str(i + 1) for i in range(nnodes)]),
    )
    return receptor_df
