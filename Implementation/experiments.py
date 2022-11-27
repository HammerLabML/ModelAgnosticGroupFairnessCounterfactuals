import warnings
warnings.filterwarnings("ignore")
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from ceml.sklearn import generate_counterfactual

from utils import load_communitiesAndCrime_dataset, load_creditCardClients_dataset, load_lawSchool_dataset
from memory_counterfactual import MemoryCounterfactual
from fair_counterfactuals import FairCounterfactualBlackBox, FairCounterfactualMemoryBlackBox



if __name__ == "__main__":
    def run_exp(verbose, memory_cf, dataset_desc, classifier_desc):
        def compute_dist(delta_cf):
            return np.sum(np.abs(delta_cf))


        def compute_counterfactual(model, x_orig, y_target):
            try:
                _, _, delta_cf = generate_counterfactual(model, x_orig, y_target, return_as_dict=False, regularization="l1", optimizer="auto")

                return delta_cf
            except Exception as ex:
                print(ex)
                return None


        def create_fainess_aware_explainer(model):
            if memory_cf is False:
                return FairCounterfactualBlackBox
            else:
                return FairCounterfactualMemoryBlackBox

        # Load data
        if dataset_desc == "creditcard":
            X, y, y_sensitive = load_creditCardClients_dataset();data_desc="creditcard"
        elif dataset_desc == "communitiescrime":
            X, y, y_sensitive = load_communitiesAndCrime_dataset();data_desc="communitiescrime"
        elif dataset_desc == "lawschool":
            X, y, y_sensitive = load_lawSchool_dataset();data_desc="lawschool"
        else:
            raise ValueError(f"Unknown dataset '{dataset_desc}'")
        if verbose:
            print(X.shape, y.shape, y_sensitive.shape)
        
        # Cross validation
        cf_dist_0_total = [];cf_dist_1_total = []
        cf_dist_fair_0_total = [];cf_dist_fair_1_total = []
        kf = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_sensitive_train, y_sensitive_test = y_sensitive[train_index], y_sensitive[test_index]
            if verbose:
                print(X_train.shape, X_test.shape)

            # Deal with imbalanced data
            sampling = RandomUnderSampler() # Undersampe majority class
            X_train, y_train = sampling.fit_resample(X_train, y_train)

            # Preprocessing
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Fit predictive model
            if classifier_desc == "logreg":
                model = LogisticRegression(multi_class="multinomial");model_desc="logreg"
            elif classifier_desc == "dectree":
                model = DecisionTreeClassifier(max_depth=7);model_desc="dectree"
            elif classifier_desc == "gnb":
                model = GaussianNB();model_desc="gnb"
            else:
                raise ValueError(f"Unknown classifier '{classifier_desc}'")
            model.fit(X_train, y_train)

            # Evaluate predictive accuracy
            if verbose:
                print(f"Train: {f1_score(y_train, model.predict(X_train))}   Test: {f1_score(y_test, model.predict(X_test))}")
                print(confusion_matrix(y_test, model.predict(X_test)))

            # Build memory counterfactual explainer
            y_pred = model.predict(X_train)
            mask = y_pred == y_train
            mem_cf_explainer = MemoryCounterfactual(X_train[mask,:], y_train[mask])

            # Evaluate group fairness of normal counterfactuals and fairness aware counterfactuals
            cf_dist_0 = [];cf_dist_1 = []
            cf_dist_fair_0 = [];cf_dist_fair_1 = []
            for i in range(X_test.shape[0]):
                x_orig = X_test[i,:]
                y_orig = y_test[i]
                y_target = 1 if y_orig == 0 else 0

                if model.predict([x_orig]) != y_orig:   # Ignore missclassified samples!
                    continue

                # Compute counterfactual
                if memory_cf is False:
                    delta_cf = compute_counterfactual(model, x_orig, y_target)
                else:
                    x_cf = mem_cf_explainer.compute_counterfactual(x_orig, y_target);delta_cf = x_cf - x_orig
                if delta_cf is None:
                    if verbose:
                        print("Computation of counterfactual failed.")
                    continue

                if y_sensitive_test[i] == 0:
                    cf_dist_0.append(compute_dist(delta_cf))
                elif y_sensitive_test[i] == 1:
                    cf_dist_1.append(compute_dist(delta_cf))

            cf_dist_0_total += cf_dist_0
            cf_dist_1_total += cf_dist_1

            if verbose:
                print("Default:")
                print(f"Group-0 (#{len(cf_dist_0)}): {np.mean(cf_dist_0)} \pm {np.std(cf_dist_0)}")
                print(f"Group-1 (#{len(cf_dist_1)}): {np.mean(cf_dist_1)} \pm {np.std(cf_dist_1)}")

            # Build fairness aware counterfactual explainer
            if memory_cf is False:
                fair_explainer = create_fainess_aware_explainer(model)(model=model, cf_dists_group_0=cf_dist_0, cf_dists_group_1=cf_dist_1);algo_desc="closest"
            else:
                fair_explainer = create_fainess_aware_explainer(model)(model=model, X_train=X_train, y_train=y_train, cf_dists_group_0=cf_dist_0, cf_dists_group_1=cf_dist_1);algo_desc="memory"

            # Same for fairness aware counterfactuals
            for i in range(X_test.shape[0]):
                x_orig = X_test[i,:]
                y_orig = y_test[i]
                y_target = 1 if y_orig == 0 else 0

                if model.predict([x_orig]) != y_orig:   # Ignore missclassified samples!
                    continue

                # Compute counterfactual
                delta_cf = fair_explainer.compute_explanation(x_orig, y_target)
                if delta_cf is None:
                    if verbose:
                        print("Computation of fair counterfactual failed.")
                    continue

                if y_sensitive_test[i] == 0:
                    cf_dist_fair_0.append(compute_dist(delta_cf))
                elif y_sensitive_test[i] == 1:
                    cf_dist_fair_1.append(compute_dist(delta_cf))
            
            cf_dist_fair_0_total += cf_dist_fair_0
            cf_dist_fair_1_total += cf_dist_fair_1

            if verbose:
                print("Fairness aware:")
                print(f"Group-0 (#{len(cf_dist_fair_0)}): {np.mean(cf_dist_fair_0)} \pm {np.std(cf_dist_fair_0)}")
                print(f"Group-1 (#{len(cf_dist_fair_1)}): {np.mean(cf_dist_fair_1)} \pm {np.std(cf_dist_fair_1)}")


        # Aggregated evaluation
        if verbose:
            print("Without fairness constraint:")
            print(f"Group-0: {np.mean(cf_dist_0_total)} \pm {np.std(cf_dist_0_total)}; {np.median(cf_dist_0_total)}")
            print(f"Female: {np.mean(cf_dist_1_total)} \pm {np.std(cf_dist_1_total)}; {np.median(cf_dist_1_total)}")

            print("With fairness constraint:")
            print(f"Group-0: {np.mean(cf_dist_fair_0_total)} \pm {np.std(cf_dist_fair_0_total)}; {np.median(cf_dist_fair_0_total)}")
            print(f"Group-1: {np.mean(cf_dist_fair_1_total)} \pm {np.std(cf_dist_fair_1_total)}; {np.median(cf_dist_fair_1_total)}")

        # Save results
        np.savez(f"results/results_{data_desc}_{model_desc}_{algo_desc}.npz", cf_dist_0=cf_dist_0_total, cf_dist_1=cf_dist_1_total, cf_fair_dist_0=cf_dist_fair_0_total, cf_fair_dist_1=cf_dist_fair_1_total)


    configurations = []
    for clf in ["logreg", "dectree", "gnb"]:
        for d in ["communitiescrime", "creditcard", "lawschool"]:
            for m in [True, False]:
                configurations.append({"verbose": False, "memory_cf": m, "dataset_desc": d, "classifier_desc": clf})

    Parallel(n_jobs=-2)(delayed(run_exp)(**config) for config in configurations)
