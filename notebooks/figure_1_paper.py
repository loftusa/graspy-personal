# %%
import math
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI
from tqdm import tqdm  # for timing loops

from graspologic.embed import CovariateAssistedEmbedding as CASE
from graspologic.embed import LaplacianSpectralEmbed as LSE
from graspologic.simulations import sbm
from graspologic.plot import heatmap, pairplot

import sys
import random
np.set_printoptions(threshold=sys.maxsize)
np.random.seed(42)

# %%


def new_labels():
    # generate array of random values
    a = np.random.rand(4500)

    # make a utility list of every position in that array, and shuffle it
    indices = [i for i in range(0, len(a))]
    random.shuffle(indices)

    # set the proportion you want to keep the same
    proportion = 0.5

    # make two lists of indices, the ones that stay the same and the ones that get shuffled
    anchors = indices[0:math.floor(len(a)*proportion)]
    not_anchors = indices[math.floor(len(a)*proportion):]

    # get values of non-anchor indices, and shuffle them
    not_anchor_values = [a[i] for i in not_anchors]
    random.shuffle(not_anchor_values)

    # loop original array, if an anchor position, keep original value
    # if not an anchor, draw value from shuffle non-anchor value list and increment the count
    final_list = []
    count = 0
    for e, i in enumerate(a):
        if e in not_anchors:
            final_list.append(i)
        else:
            final_list.append(not_anchor_values[count])
            count += 1

    # test proportion of matches and non-matches in output

    match = []
    not_match = []
    for e, i in enumerate(a):
        if i == final_list[e]:
            match.append(True)
        else:
            not_match.append(True)
    len(match)/(len(match)+len(not_match))

    return
# %%


def gen_sbm(p, q, assortative=True, N=1500):
    if not assortative:
        p, q = q, p

    n = N//3
    A = np.full((3, 3), q)
    A[np.diag_indices_from(A)] = p
    return sbm([n, n, n], A, return_labels=True)


def gen_covariates(labels, m1, m2, agreement=1, d=3):
    """
    n x 3 matrix of covariates

    """
    n_total = len(labels)
    m2_array = np.random.choice([1, 0], p=[m2, 1-m2], size=(n_total, d))
    m1_array = np.random.choice([1, 0], p=[m1, 1-m1], size=n_total)
    m2_array[np.arange(n_total), labels] = m1_array
    return m2_array


# p = 0.5
# array = np.repeat([0, 1, 2], 2)
# print(array)
# N = len(array)
# new_array = array.copy()

# indx = np.random.choice(int(N), int(N*p), replace=False)
# print(indx)
# shuf_indx = np.random.permutation(indx)
# print(shuf_indx)

# new_array[shuf_indx] = array[indx]
# print(new_array)
# print(np.count_nonzero(array - new_array)/N)
# print(new_array)

# labels
# np.count_nonzero(labels_ - labels) / len(labels)
# X = gen_covariates(labels, m1=.8, m2=.2, agreement=agreement)

# fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# heatmap(A, ax=axs[0])
# sns.heatmap(X, ax=axs[1])
# %%


def get_misclustering(A, model, labels, covariates=None) -> float:
    if covariates is None:
        Xhat = model.fit_transform(A)
    elif covariates == "generate":
        pass  # TODO
    else:
        Xhat = model.fit_transform(A, covariates=covariates)

    kmeans = KMeans(n_clusters=3)
    labels_ = kmeans.fit_predict(Xhat)
    misclustering = 1 - ARI(labels, labels_)

    return misclustering


def trial(p=.03, q=.015, m1=.8, m2=.02, agreement=1, assort=True, algs=["assortative", "non_assortative", "CCA", "LSE", "COV"]) -> dict:
    """
    Return misclustering rates for all models under particular assumptions.

    """

    # set up models
    nc = 3
    assrttv_model = CASE(embedding_alg="assortative",
                         n_components=nc, normalize=True)
    non_assrttv_model = CASE(embedding_alg="non-assortative",
                             n_components=nc, normalize=True)
    cca_model = CASE(embedding_alg="cca", n_components=nc, normalize=True)
    reg_LSE_model = LSE(form="R-DAD", n_components=nc,
                        normalize=True, check_lcc=False)
    cov_LSE_model = clone(reg_LSE_model)

    # collect models
    casc_models = {"assortative": assrttv_model,
                   "non_assortative": non_assrttv_model,
                   "CCA": cca_model,
                   "LSE": reg_LSE_model,
                   "COV": cov_LSE_model}

    # for testing purposes
    casc_models = {name: casc_models[name] for name in algs}

    # generate data
    N = 1500*3
    A, labels = gen_sbm(p, q, assortative=assort, N=N)
    X = gen_covariates(labels, m1, m2, agreement=agreement)  # TODO

    # fit, cluster, get misclustering rates
    misclusterings = {}
    for name, model in casc_models.items():
        if name in {"assortative", "non_assortative", "CCA"}:
            misclustering = get_misclustering(A, model, labels, covariates=X)
        elif name == "LSE":
            misclustering = get_misclustering(A, model, labels)
        elif name == "COV":
            misclustering = get_misclustering(X@X.T, model, labels)

        misclusterings[name] = misclustering

    # return misclustering rates
    return misclusterings


def trials(p=.03, q=.015, m1=.8, m2=.02, trial_type="", agreement=1, assortative=True):
    """
    vary within-minus between-block probability (p-q)
    """
    num_trials = 6
    algorithms = ["assortative", "non_assortative", "CCA", "LSE", "COV"]
    # algorithms = ["LSE"]

    # set trial parameters
    if trial_type == "probability":
        max_diff = .025
        x, y = p, q
    elif trial_type == "covariate":
        max_diff = .6
        x, y = m1, m2
    else:
        raise ValueError("need trial_type")

    # generate test space
    xs = np.full(num_trials, y)
    diffs = np.linspace(0, max_diff, num=num_trials)
    ys = xs + diffs
    probs = np.c_[xs, ys]

    # trials
    results = np.zeros((num_trials, len(algorithms)+1))
    results[:, 0] = diffs
    for i, (x, y) in tqdm(enumerate(probs)):
        # for i, (p, q) in enumerate(probs):
        if trial_type == "probability":
            misclusterings = trial(
                p=x, q=y, assort=assortative, algs=algorithms)
        elif trial_type == "covariate":
            misclusterings = trial(
                m1=x, m2=y, assort=assortative, algs=algorithms)
        elif trial_type == "membership":
            misclusterings = trial(agreement=agreement,
                                   assort=assortative, algs=algorithms)
        for j, name in enumerate(misclusterings.keys()):
            j += 1  # to account for the diffs column
            results[i, j] = misclusterings[name]

    columns = ["diffs"] + list(misclusterings.keys())
    results = pd.DataFrame(data=results, columns=columns)
    return results


def plot_results(results, ax=None, xlabel="", title=""):
    if ax is None:
        ax = plt.gca()

    linetypes = {"assortative": "k--", "non_assortative": "k-",
                 "CCA": "k:", "LSE": "k-."}

    X = results.loc[:, "diffs"].values
    results_ = results.drop("diffs", axis=1)
    for name in results_.columns:
        if name == "COV":
            # custom linetype to get long dashes
            line, = ax.plot(X, results_["COV"], 'k', label=name)
            line.set_dashes([15, 5])
        else:
            line, = ax.plot(X, results_[name], linetypes[name], label=name)

    ax.set(xlabel=xlabel, ylabel="Average misclustering rate",
           title=title, ylim=(0, 1.2))

    return line


# make figure
fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(
    10, 15), constrained_layout=True)
assortative_title = "Assortative graph, varying graph"
non_assortative_title = "Non-assortative graph, varying graph"

# plot probability trials
#  assortative
xlabel = "Within- minus between-block probability (p-q)"
# assortative_prob = trials(trial_type="probability", assortative=True)
plot_results(assortative_prob,
             ax=axs[0, 0], xlabel=xlabel, title=assortative_title)

#  non-assortative
# non_assortative_prob = trials(trial_type="probability", assortative=False)
plot_results(non_assortative_prob,
             ax=axs[0, 1], xlabel=xlabel, title=non_assortative_title)

# plot covariate trials
#  assortative
xlabel = "Difference in covariate probabilities (m1 - m2)"
# assortative_cov = trials(trial_type="covariate", assortative=True)
plot_results(assortative_cov,
             ax=axs[1, 0], xlabel=xlabel, title=assortative_title)

#  non-assortative
# non_assortative_cov = trials(trial_type="covariate", assortative=False)
plot_results(non_assortative_cov,
             ax=axs[1, 1], xlabel=xlabel, title=non_assortative_title)

# figure legend
handles, labels = axs[0, 0].get_legend_handles_labels()
labels = [i if i != "COV" else "COV (long dash)" for i in labels]
fig.legend(handles, labels, loc=(.81, .9))

plt.savefig("../figs/figure1_paper.png", bbx_inches="tight")
# %%
# TODO

# xlabel = "Covariate to graph block membership agreement"
# axs[2, 0].set_xlabel(xlabel)
# axs[2, 1].set_xlabel(xlabel)
