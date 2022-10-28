"""
Functions for performing regression analysis between connectivity features and cognition
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, GridSearchCV
from scipy import stats


def get_module_img(atlas, rois, value=1):
    atlas_data = atlas.get_fdata()
    module_img = np.where(atlas_data == (np.array(rois) + 1), value, 0).sum(axis=3)
    return module_img


def get_mae_cv(X, y, model, folds=10):
    Y_pred = cross_val_predict(model, X, y, cv=folds)
    # reg = model.fit(X, y)
    # Y_pred = reg.predict(X)
    mae = mean_absolute_error(y, Y_pred)
    mae_std = np.std(np.abs(Y_pred - y))
    return mae, mae_std


def multivariate_feature_ranking(df, y, model, folds=10):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    selector = RFECV(model, step=1, cv=folds, scoring="neg_mean_absolute_error")
    selector.fit(X, y)
    X_sorted = X[:, selector.ranking_ - 1]
    names_sorted = df.columns[selector.ranking_ - 1]
    return X_sorted, names_sorted


def univariate_feature_ranking(df, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    ft_corr = abs(np.corrcoef(X.T, y)[0 : (len(X.T)), len(X.T)])
    X_sorted = X[:, np.argsort(ft_corr)[::-1]]
    r_vals_sorted = ft_corr[np.argsort(ft_corr)[::-1]]
    names_sorted = df.columns[np.argsort(ft_corr)[::-1]]
    return X_sorted, names_sorted, r_vals_sorted


def iterative_regression(X, y, nfeat, model, folds=10):
    mae_m = []
    mae_s = []
    for idx in range(1, nfeat):
        X_sub_samp = X[:, :idx]
        mae_mean, mae_std = get_mae_cv(X_sub_samp, y, model, folds)
        mae_m.append(mae_mean)
        mae_s.append(mae_std)
    return mae_m, mae_s


def optimal_modules(X, y, model, mae, names_sorted):
    min_mae_position = np.where(mae == np.min(mae))[0][0]
    model_optimal = model.fit(X[:, : min_mae_position + 1], y)
    optimal_coeffs = model_optimal.coef_[0]
    optimal_modules = names_sorted[: min_mae_position + 1]
    return optimal_coeffs, optimal_modules


def optimal_brain_map(
    project_path, atlas_name, optimal_modules, optimal_coeffs, label_dict
):
    atlas = nib.load(os.path.join(project_path, atlas_name))
    module_coef = []
    for idx, mod in enumerate(optimal_modules):
        rois = label_dict[
            mod[6:]
        ]  # [6:] is to remove FCINT_/FCEXT_ from the module name
        module_coef.append(get_module_img(atlas, rois, abs(optimal_coeffs[idx])))

    coeff_module_avg = np.array(module_coef).sum(axis=0)
    coeff_module_img = nib.Nifti1Image(coeff_module_avg, affine=atlas.affine)
    return coeff_module_img
