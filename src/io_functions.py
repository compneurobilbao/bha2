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


def load_data_sex_paired(data_path, conn_size, conn_type, sex_column_name):
    participants_df = pd.read_csv(data_path + "/participants.tsv", sep="\t")
    np.random.seed(42)
    participants_df = pd.read_csv(data_path + "/participants.tsv", sep="\t")
    sex = participants_df[sex_column_name].values
    sex_classes = np.unique(sex)

    if np.count_nonzero(sex == sex_classes[0]) > np.count_nonzero(
        sex == sex_classes[1]
    ):
        filtered_participants = pd.concat(
            [
                participants_df[participants_df[sex_column_name] == sex_classes[1]],
                participants_df[
                    participants_df[sex_column_name] == sex_classes[0]
                ].sample(n=np.count_nonzero(sex == sex_classes[1])),
            ]
        ).reset_index(drop=True)
    elif np.count_nonzero(sex == sex_classes[1]) > np.count_nonzero(
        sex == sex_classes[0]
    ):
        filtered_participants = pd.concat(
            [
                participants_df[participants_df[sex_column_name] == sex_classes[0]],
                participants_df[
                    participants_df[sex_column_name] == sex_classes[1]
                ].sample(n=np.count_nonzero(sex == sex_classes[0])),
            ]
        ).reset_index(drop=True)
    else:
        raise ValueError("More than two classes in sex column")

    files = []
    for id in filtered_participants["ID"]:
        pattern = (
            data_path
            + "/iPA_"
            + str(conn_size)
            + "/"
            + conn_type
            + "/"
            + "*"
            + str(id)
            + "*.csv"
        )
        matching_files = glob.glob(pattern)
        files.extend(matching_files)

    matrices = np.array(
        [
            pd.read_csv(
                f, header=None, delimiter=" ", dtype=np.float16, engine="c"
            ).to_numpy(dtype=np.float16)
            for f in sorted(files)
        ]
    )
    return matrices
