"""
Functions for calculate connectivity features based on dedrogram tree
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import json

def connectome_average(slist):
    sc_all = []
    fc_all = []
    for i, sub in enumerate(slist):
        print(i)
        sc = np.array(pd.read_csv('sc_matrices/' + sub + '_anat_probabilistic_connectome.csv', delimiter=' ', header=None))
        fc = np.corrcoef(np.transpose(np.genfromtxt('timeseries/ts_' + sub + '.txt')))
        scmod = np.log10(sc+1)
        fc_all.append(fc)
        sc_all.append(scmod/scmod.max())
    return sc_all, fc_all

def modularity(A, T):
    m = np.sum(A)/2
    n = A.shape[0]
    Q = 0
    for i in range(n):
        for j in range(n):
            if T[i] == T[j]:
                Q = Q + (A[i,j] - (np.sum(A[i,:])*np.sum(A[j,:]))/(2*m))/(2*m)
    return Q

def mod_similarity(fc, sc, T):
    fc_mod_m = []
    sc_mod_m = []

    for i in range(1, np.max(T)+1):
        rois_in_clust = np.where(T == i)[0]
        fcm_mod = fc[rois_in_clust,:]
        fcm_mod = fcm_mod[:,rois_in_clust]
        scm_mod = sc[rois_in_clust,:]
        scm_mod = scm_mod[:,rois_in_clust]
        fc_mod_m.append(fcm_mod.mean())
        sc_mod_m.append(scm_mod.mean())

    return np.sqrt((np.multiply(fc_mod_m, sc_mod_m)).mean())

def calc_connfeat(fc, sc, T, lvl):
    features = np.array([])
    feat_names = np.array([])
    feat_dict = {}

    for i in range(1, np.max(T)+1):
        rois_in_clust = np.where(T == i)[0]
        ext_rois = np.setdiff1d(np.array([i for i in range(len(T))]), rois_in_clust)

        if len(rois_in_clust) > 1:
            desc = ('lvl_' + str(lvl) + '_mod_' + str(i))
            feat_dict[desc] = list(rois_in_clust)

            fc_int = fc.iloc[rois_in_clust][rois_in_clust].to_numpy().mean()
            fc_out = fc.iloc[rois_in_clust][ext_rois].to_numpy().mean()
            sc_int = (sc.iloc[rois_in_clust][rois_in_clust].to_numpy().sum())/len(rois_in_clust)
            sc_out = (sc.iloc[rois_in_clust][ext_rois].to_numpy().sum())/len(rois_in_clust)

            features = np.concatenate((features,
            np.array([fc_int, fc_out, sc_int, sc_out])))
            feat_names = np.concatenate((feat_names,
                np.array(['FCINT_' + desc, 'FCEXT_' + desc, 'SCINT_' + desc, 'SCEXT_' + desc])))


    return features, feat_names, feat_dict

def generate_population_features(slist, g, scm, fcm):
    cc = np.multiply(((g*abs(fcm)) + ((1-g) * scm)), np.sign(fcm))
    cc_dist = pdist(cc, 'cosine')
    W = cc_dist/max(cc_dist)
    Z = linkage(W, 'weighted')

    Xfeatures = slist.reshape(len(slist), 1)
    Xnames = np.array(['label'])
    Xdesc_dict = {}
    for numClust in range(10,20,1):
        T = fcluster(Z, numClust, criterion='maxclust')
        pfeatures = []

        for i, sub in enumerate(slist):
            print('gamma = ' + str(g) + ', lvl = ' + str(numClust) + ', snumber = ' + str(i))
            sc = np.array(pd.read_csv('sc_matrices/' + sub + '_anat_probabilistic_connectome.csv', delimiter=' ', header=None))
            fc = np.corrcoef(np.transpose(np.genfromtxt('timeseries/ts_' + sub + '.txt')))
            scmod = np.log10(sc+1)
            subfeatures, subfeat_names, feat_dict = calc_connfeat(fc, scmod/scmod.max(), T, numClust)
            pfeatures.append(subfeatures)

        Xfeatures = np.hstack((Xfeatures, np.array(pfeatures)))
        Xnames = np.hstack((Xnames, np.array(subfeat_names)))
        Xdesc_dict.update(feat_dict)


    Xdf = pd.DataFrame(Xfeatures)
    Xdf.columns = Xnames
    Xdf = Xdf.loc[:,~Xdf.apply(lambda x: x.duplicated(),axis=1).all()].copy()
    return Xdf, Xdesc_dict
