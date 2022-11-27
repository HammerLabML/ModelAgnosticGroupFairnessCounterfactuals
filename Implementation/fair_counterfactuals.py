from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp
from ceml.sklearn import generate_counterfactual


class FairCounterfactual(ABC):
    def __init__(self, model, cf_dists_group_0, cf_dists_group_1):
        self.model = model
        self.cf_dists_group_0 = cf_dists_group_0
        self.cf_dists_group_1 = cf_dists_group_1
        self.cf_dist_mean = None
        self.cf_dist_std = None

        # Compute statistics of disadvantaged group
        mean_0, std_0, median_0 = np.mean(self.cf_dists_group_0), np.std(self.cf_dists_group_0), np.median(self.cf_dists_group_0)
        mean_1, std_1, median_1 = np.mean(self.cf_dists_group_1), np.std(self.cf_dists_group_1), np.median(self.cf_dists_group_1)

        if mean_0 > mean_1 or median_0 > median_1:  # Group 0 is disadvantaged
            self.cf_dist_mean = mean_0
            self.cf_dist_std = std_0
        elif mean_1 > mean_0 or median_1 > median_0:    # Group 1 is disadvantaged
            self.cf_dist_mean = mean_1
            self.cf_dist_std = std_1
        else:   # No one is disadvantaged!
            self.cf_dist_mean = mean_0
            self.cf_dist_std = std_0

    def _compute_cf_dist(self, random):
        if random is False:
            return self.cf_dist_mean
        else:
            return np.abs(np.random.normal(self.cf_dist_mean, self.cf_dist_std))

    def compute_dist(self, x_orig, x_cf):
        return np.sum(np.abs(x_orig - x_cf))

    @abstractmethod
    def compute_explanation(self, x_orig, y_target):
        raise NotImplementedError()


class FairCounterfactualMemoryBlackBox(FairCounterfactual):
    def __init__(self, X_train, y_train, **kwds):
        super().__init__(**kwds)

        self.X_train = []
        self.y_train_pred = []
        for idx in range(X_train.shape[0]):
            y_pred = self.model.predict([X_train[idx, :]])
            if y_pred == y_train[idx]:
                self.X_train.append(X_train[idx, :])
                self.y_train_pred.append(y_pred)
        self.X_train = np.array(self.X_train)
        self.y_train_pred = np.array(self.y_train_pred).flatten()

    def compute_explanation(self, x_orig, y_target, random=True):
        # Sample the minimum distance/cost
        cf_dist_min = self._compute_cf_dist(random)

        # Select a suitable sample from the traning set
        x_cf = None
        cur_best_dist = None
        mask = self.y_train_pred == y_target
        for x in self.X_train[mask,:]:
            d = self.compute_dist(x_orig, x)
            if d >= cf_dist_min:
                if cur_best_dist == None or d < cur_best_dist:
                    cur_best_dist = d
                    x_cf = x

        return x_cf


class FairCounterfactualBlackBox(FairCounterfactual):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def compute_explanation(self, x_orig, y_target, C_pred=100., C_fair=1000., random=True):
        # Define loss function
        cf_dist_min = self._compute_cf_dist(random)

        def loss(xcf):
            return np.linalg.norm(x_orig - xcf, 1) + C_pred * int(self.model.predict([xcf]) != y_target) + C_fair * max(cf_dist_min - self.compute_dist(x_orig, xcf), 0)

        # Find a good starting point
        x0, _, _ = generate_counterfactual(self.model, x_orig, y_target, return_as_dict=False, regularization="l1", optimizer="auto")

        # Minimize loss function
        r = minimize(loss, x0, method="Nelder-Mead")

        # Sanity check solution
        xcf = r.x
        delta_cf = x_orig - xcf

        if self.model.predict([xcf]) != y_target:   # Invalid counterfactual!
            return None
        else:
            return delta_cf
