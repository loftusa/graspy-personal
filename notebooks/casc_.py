# %%
# gonna do this in code as I read

from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.embed.svd import selectSVD
from graspologic.embed.base import BaseSpectralEmbed
import numpy as np
import graspologic as gs
from graspologic.simulations import sbm
from graspologic.plot import heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%


def gen_covariates(m1, m2, labels):
    # TODO: make sure labels is 1d array-like
    n = len(labels)
    m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
    m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, 3))
    m2_arr[np.arange(n), labels] = m1_arr
    return m2_arr


# %%

n = 200
n_communities = 3
p, q = 0.9, 0.3
B = np.array([[p, q, q], [q, p, q], [q, q, p]])

B2 = np.array([[q, p, p], [p, q, p], [p, p, q]])

A, labels = sbm([n, n, n], B, return_labels=True)
N = A.shape[0]
L = gs.utils.to_laplace(A, form="R-DAD")
X = gen_covariates(0.9, 0.1, labels)


heatmap(L)
# heatmap(L)
# %%
sns.heatmap(X)
# %%


# %%

# A good initial choice of a is the value which makes the leading eigenvalues of LL and aXX^T equal, namely
# a_0 = \lambda_1 (LL) / \lambda_1 (XX^T)
Lsquared = L @ L
Lleading = sorted(np.linalg.eigvals(Lsquared), reverse=True)[0]
Xleading = sorted(np.linalg.eigvals(X @ X.T), reverse=True)[0]
a = np.float(Lleading / Xleading)
L_ = (L @ L) + (a * (X @ X.T))
# heatmap(L_)
# heatmap(Lsquared)
# heatmap(X @ X.T)

ase = AdjacencySpectralEmbed(n_components=2)
ase._reduce_dim(L)
X = ase.latent_left_

scatter = plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.gcf().set_size_inches(5, 5)
ax = plt.gca()
ax.legend(*scatter.legend_elements())
plt.xlim(-.05, .05)
plt.ylim(-.05, .05)
plt.title(r"Spectral embedding of $LL + aXX^T$")
# plt.savefig(
#     "/Users/alex/Dropbox/School/NDD/graspy-personal/figs/casc_working.png"
# )

# %%
heatmap(L_)
plt.title(r"$LL + aXX^T$")
# heatmap(X@X.T)

plt.savefig(
    "/Users/alex/Dropbox/School/NDD/graspy-personal/figs/L_.png", bbox_inches="tight")
