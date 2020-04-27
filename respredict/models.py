import numpy as np
from sklearn.kernel_approximation import RBFSampler
import sklearn.linear_model
#import fastfood

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline



import time

class RBFRidge(object):
    def __init__(self, alpha=0.1, gamma=0.1, 
                 n_components=1024, seed=0, normalize=True):

        self.rbf_feature = RBFSampler(gamma=gamma, random_state=seed, 
                                    n_components=n_components)

        self.normalize = sklearn.preprocessing.StandardScaler()
        
        if normalize:
            self.preproc = Pipeline([('normalize', self.normalize), 
                                     ('rbf', self.rbf_feature)])
        else:
            self.preproc = Pipeline([('rbf', self.rbf_feature)])



        self.m = sklearn.linear_model.Ridge(alpha=alpha)

    
    def fit(self, X, y):
        t1 = time.time()
        X_transformed = self.preproc.fit_transform(X)
        t2 = time.time()
        self.m.fit(X_transformed, y)
        t3 = time.time()
        print("RBF Features took {:3.1f}s, fit took {:3.1f}s".format(t2-t1, t3-t2))
        return self

    def predict(self, X):
        t1 = time.time()
        X_transformed = self.preproc.transform(X)
        t2 = time.time()
        y = self.m.predict(X_transformed)
        t3 = time.time()
        print("predict of {} : RBF Features took {:3.1f}s, pred took {:3.1f}s".format(X.shape, 
                                                                                      t2-t1, t3-t2))

        return y 
        

class FFRidge(object):
    """
    Fastfood-based
    """
    def __init__(self, alpha=0.1, scale=0.1, 
                 n_components=1024, seed=0):

        self.n_components = n_components

        self.ff_feature = fastfood.Fastfood(scale=scale,
                                            n_components=n_components)

        self.m = sklearn.linear_model.Ridge(alpha=alpha)

    
    def fit(self, X, y):
        t1 = time.time()
        X_transformed = self.ff_feature.fit_transform(X).astype(np.float64)
        t2 = time.time()
        fit_res = self.m.fit(X_transformed, y)
        t3 = time.time()
        print("ff took {:3.1f}s, fit took {:3.1f}s".format(t2-t1, t3-t2))
        print(self.m)

        return self.m

    def predict(self, X):

        X_transformed = self.ff_feature.transform(X).astype(np.float64)
        return self.m.predict(X_transformed)
        

