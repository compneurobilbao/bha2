"""Definintion of functions related to load the connectomes."""

import pandas as pd
import numpy as np
import glob


def load_data(path):
    """
    Load data from multiple CSV files in a directory.

    Parameters
    ----------
    path : str
        The path to the directory containing the connectome files.

    Returns:
    -------
    matrices : ndarray, shape (N, M, M)
        An array of matrices, where each matrix represents the connectome of each participant.
    """
    # List all the files in the directory
    files = glob.glob(path + "/*.csv")
    # Load all the files in the directory
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
    """
    Load data from multiple connectomes, paired by the sex of the participants

    Parameters
    ----------
    data_path : str
        The path to the data folder.
    conn_size : int
        The size of the connectome.
    conn_type : str
        The type of the connectome (could be sc or fc).
    sex_column_name : str
        The name of the sex column in participants.tsv file.

    Returns:
    -------
    matrices : ndarray, shape (N, M, M)
        An array of matrices, where each matrix represents the connectome of each participant.
    """
    # Loading the participants file
    participants_df = pd.read_csv(data_path + "/participants.tsv", sep="\t")
    # Getting the sex of the participants
    sex = participants_df[sex_column_name].values
    sex_classes = np.unique(sex)

    # The removing of the bigger class is doing randomly
    np.random.seed(42)

    # Removing the bigger class
    if np.count_nonzero(sex == sex_classes[0]) > np.count_nonzero(
        sex == sex_classes[1]
    ):
        # Concatenating the smaller class with a random sample of the bigger class
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
        # Concatenating the smaller class with a random sample of the bigger class
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

    # The files to load are the ones in the new filtered participants dataframe
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
