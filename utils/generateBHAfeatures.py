import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

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
        roisInClust = np.where(T == i)[0]
        fcm_mod = fc[roisInClust,:]
        fcm_mod = fcm_mod[:,roisInClust]
        scm_mod = sc[roisInClust,:]
        scm_mod = scm_mod[:,roisInClust]
        fc_mod_m.append(fcm_mod.mean())
        sc_mod_m.append(scm_mod.mean())
    
    return np.sqrt((np.multiply(fc_mod_m, sc_mod_m)).mean())
    
def calc_connfeat(fc, sc, T, lvl):
    features = np.array([])
    feat_names = np.array([])
    feat_desc = []

    for i in range(1, np.max(T)+1):
        roisInClust = np.where(T == i)[0]
        extRois = np.ones(fc.shape[0], dtype=bool)
        extRois[roisInClust] = False
        if len(roisInClust) > 1:
            desc = ('lvl_' + str(lvl) + '_mod_' + str(i))
            feat_desc.append([desc, roisInClust])
            
            fcm_int = fc[roisInClust,:]
            fcm_int = fcm_int[:,roisInClust]
            scm_int = sc[roisInClust,:]
            scm_int = scm_int[:,roisInClust]
            fcm_ext = fc[roisInClust,:]
            fcm_ext = fcm_ext[:,extRois]
            scm_ext = sc[roisInClust,:]
            scm_ext = scm_ext[:,extRois]

            features = np.concatenate((features, 
            np.array([fcm_int.mean(), fcm_ext.mean(), scm_int.mean(), scm_ext.mean()])))
            feat_names = np.concatenate((feat_names, 
                np.array(['FCINT_' + desc, 'FCEXT_' + desc, 'SCINT_' + desc, 'SCEXT_' + desc]))) 


    return features, feat_names, feat_desc

def generate_features(slist, g, scm, fcm):
    cc = np.multiply(((g*abs(fcm)) + ((1-g) * scm)), np.sign(fcm))
    cc_dist = pdist(cc, 'cosine')
    W = cc_dist/max(cc_dist)
    Z = linkage(W, 'weighted')

    Xfeatures = slist.reshape(len(slist), 1)
    Xnames = np.array(['label'])
    Xdesc = []
    for numClust in range(10,20,1):
        T = fcluster(Z, numClust, criterion='maxclust')
        pfeatures = []

        for i, sub in enumerate(slist):
            print('gamma = ' + str(g) + ', lvl = ' + str(numClust) + ', snumber = ' + str(i))
            sc = np.array(pd.read_csv('sc_matrices/' + sub + '_anat_probabilistic_connectome.csv', delimiter=' ', header=None))
            fc = np.corrcoef(np.transpose(np.genfromtxt('timeseries/ts_' + sub + '.txt')))
            scmod = np.log10(sc+1)
            subfeatures, subfeat_names, feat_desc = calc_connfeat(fc, scmod/scmod.max(), T, numClust)
            pfeatures.append(subfeatures)
                        
        Xfeatures = np.hstack((Xfeatures, np.array(pfeatures)))
        Xnames = np.hstack((Xnames, np.array(subfeat_names)))
        Xdesc.append(feat_desc)
                
    
    Xdf = pd.DataFrame(Xfeatures)
    Xdf.columns = Xnames
    Xdf = Xdf.loc[:,~Xdf.apply(lambda x: x.duplicated(),axis=1).all()].copy()
    return Xdf, Xdesc