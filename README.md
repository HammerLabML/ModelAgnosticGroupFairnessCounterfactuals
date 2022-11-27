# "Explain it in the Same Way!" -- Model-Agnostic Group Fairness of Counterfactual Explanations

This repository contains the implementation of the methods proposed in the paper ["Explain it in the Same Way!" -- Model-Agnostic Group Fairness of Counterfactual Explanations](paper.pdf) by André Artelt and Barbara Hammer.

The experiments, as described in the paper, are implemented in the folder [Implementation/](Implementation/).

## Abstract

Counterfactual explanations are a popular type of explanation for making the outcomes of a decision making system transparent to the user. Counterfactual explanations tell the user what to do in order to change the outcome of the system in a desirable way. However, it was recently discovered that the recommendations of what to do can differ significantly in their complexity between protected groups of individuals. Providing more difficult recommendations of actions to one group leads to a disadvantage of this group compared to other groups.

In this work we propose a model-agnostic method for computing counterfactual explanations that do not differ significantly in their complexity between protected groups.

## Details

### Data

The data sets used in this work are stored in [Implementation/data/](Implementation/data/). Note that many of thise .csv files in the data folder were downloaded from https://github.com/tailequy/fairness_dataset/tree/main/experiments/data.

### Implementation of the proposed method and the experiments

Algorithm 1 for computing group fair counterfactuals is implemeneted in [Implementation/fair_counterfactuals.py](Implementation/fair_counterfactuals.py) and all experiments are implemented in [Implementation/experiments.py](Implementation/experiments.py).

A minimalistic example how to use our implemenation of Algorithm 1 is given below:
```python
model = ...     # Classifier h() that is going to be explained
x_orig = ...    # Input for which we want to find a counterfactual x_cf 
y_target = ...  # Requested prediction y_cf = h(x_cf)

cf_dist_0 = ...  # Distances/Costs of counterfactuals from first group
cf_dist_1 = ...  # Distances/Costs of counterfactuals from second group

# Generating group fair counterfactual explanations
expl = FairCounterfactualBlackBox(model=model, cf_dists_group_0=cf_dist_0, cf_dists_group_1=cf_dist_1)
delta_cf = expl.compute_explanation(x_orig, y_target)  # Compute explanation
x_cf = x_orig + delta_cf
```

## Requirements

- Python3.8
- Packages as listed in [Implementation/REQUIREMENTS.txt](Implementation/REQUIREMENTS.txt)


## License

MIT license - See [LICENSE](LICENSE).

## How to cite

You can cite the version on [arXiv]().
