import numpy as np


class MemoryCounterfactual():
    def __init__(self, X, y, norm=1):
        self.X = X
        self.y = y
        self.dist = self.__build_norm(norm)

    def __build_norm(self, norm_desc):
        return lambda x: np.linalg.norm(x, ord=norm_desc)

    def compute_counterfactual(self, x_orig, y_target):
        mask = self.y == y_target
        X_ = self.X[mask,:]
        
        X_diff = X_ - x_orig
        dist = [self.dist(X_diff[i,:].flatten()) for i in range(X_diff.shape[0])]
        idx = np.argmin(dist)

        return X_[idx,:]
