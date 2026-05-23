"""
metrics.py — Cross-modal alignment metrics.

Functions
---------
linear_cka            : Linear CKA similarity (↑ better)
coral_distance        : CORAL covariance distance (↓ better)
mmd                   : Maximum Mean Discrepancy with RBF kernel (↓ better)
svcca                 : SVCCA similarity via truncated SVD + CCA (↑ better)
relative_modality_gap : Relative modality gap (↓ better)
spectral_alignment_score           : Directional SAS X→Y (↑ better)
symmetric_spectral_alignment_score : Symmetric SAS = 0.5*(SAS(X→Y)+SAS(Y→X)) (↑ better)
centroid_distances    : Euclidean and cosine distance between modality centroids
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cross_decomposition import CCA


# ── Linear CKA ────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """
    Linear CKA between X and Y (mean-centred internally).
    Returns a scalar in [0, 1].
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic  = torch.norm(X.T @ Y, p="fro") ** 2
    norm_x = torch.norm(X.T @ X, p="fro")
    norm_y = torch.norm(Y.T @ Y, p="fro")
    return hsic / (norm_x * norm_y)


# ── CORAL ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def coral_distance(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Frobenius distance between the covariance matrices of X and Y."""
    def covariance(Z):
        Zc = Z - Z.mean(dim=0, keepdim=True)
        cov = (Zc.T @ Zc) / (Zc.size(0) - 1)
        return cov + eps * torch.eye(cov.size(0), device=Z.device, dtype=Z.dtype)

    return torch.linalg.norm(covariance(X) - covariance(Y), ord="fro")


# ── MMD ───────────────────────────────────────────────────────────────────────

_MMD_SIGMA = 10.0
_MMD_SCALE = 100


def mmd(x, y):
    """
    Biased MMD with RBF kernel (σ=10).
    Implements Eq. (5) of Gretton et al., JMLR 2012.
    """
    x_sq = torch.diag(x @ x.T)
    y_sq = torch.diag(y @ y.T)
    gamma = 1 / (2 * _MMD_SIGMA ** 2)
    k_xx = torch.mean(torch.exp(-gamma * (-2 * x @ x.T + x_sq[:, None] + x_sq[None, :])))
    k_xy = torch.mean(torch.exp(-gamma * (-2 * x @ y.T + x_sq[:, None] + y_sq[None, :])))
    k_yy = torch.mean(torch.exp(-gamma * (-2 * y @ y.T + y_sq[:, None] + y_sq[None, :])))
    return _MMD_SCALE * (k_xx + k_yy - 2 * k_xy)


# ── SVCCA ─────────────────────────────────────────────────────────────────────

def svcca(feats_A, feats_B, cca_dim=10):
    """SVCCA similarity: truncated SVD followed by CCA."""
    def _preprocess(act):
        act = act - act.mean(dim=0)
        return act / (act.std(dim=0) + 1e-8)

    feats_A = _preprocess(feats_A)
    feats_B = _preprocess(feats_B)

    U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
    U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
    U1 = U1.cpu().detach().numpy()
    U2 = U2.cpu().detach().numpy()

    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    U1_c, U2_c = cca.transform(U1, U2)

    # Small jitter to avoid NaN in corrcoef for near-constant components.
    U1_c += 1e-10 * np.random.randn(*U1_c.shape)
    U2_c += 1e-10 * np.random.randn(*U2_c.shape)

    return float(np.mean(
        [np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)]
    ))


# ── Relative Modality Gap ─────────────────────────────────────────────────────

def relative_modality_gap(X, Y):
    """Relative modality gap on the unit hypersphere."""
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)
    n = X.shape[0]

    xy = (1 - F.cosine_similarity(X, Y)).mean()

    xx = 1 - X @ X.T
    xx = 1 - torch.tril(xx, diagonal=-1)
    xx = xx[xx != 1].sum()

    yy = 1 - Y @ Y.T
    yy = torch.tril(yy, diagonal=-1)
    yy = yy[yy != 0].sum()

    denominator = (1 / (2 * n * (n - 1))) * (xx + yy) + xy
    return xy / denominator


# ── Spectral Alignment Score ──────────────────────────────────────────────────

def spectral_alignment_score(X, Y, topq=0.1):
    """
    Directional SAS: how well Y's variance aligns with X's principal directions.

    X, Y  : (n, d) tensors (image and text embeddings respectively)
    topq  : fraction of X's eigenspectrum to consider (default: top 10%)
    Returns a scalar in [0, 1].
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    K_X = X.T @ X / X.shape[0]
    X_evals, X_evecs = torch.linalg.eigh(K_X)
    X_evals = X_evals.flip(0)
    X_evecs = X_evecs.flip(1)

    Z_x = X @ X_evecs
    Z_y = Y @ X_evecs

    A    = Z_x.T @ Z_y / Y.shape[0]
    diag = torch.diag(A)
    var_y = Z_y.var(dim=0, unbiased=False)
    rho  = diag / torch.sqrt(X_evals * var_y + 1e-8)

    lambda_thresh = torch.quantile(X_evals, 1 - topq)
    mask  = X_evals >= lambda_thresh
    score = (X_evals[mask] * rho[mask].abs()).sum() / X_evals[mask].sum()
    return score.clamp(0, 1)


def symmetric_spectral_alignment_score(X, Y, topq=0.1):
    """Symmetric SAS = 0.5 * (SAS(X→Y) + SAS(Y→X))."""
    return 0.5 * (spectral_alignment_score(X, Y, topq) + spectral_alignment_score(Y, X, topq))


# ── Centroid Distances ────────────────────────────────────────────────────────

def centroid_distances(X, Y):
    """Euclidean and cosine distance between the centroids of X and Y."""
    mu_x, mu_y = X.mean(0), Y.mean(0)
    euclidean = torch.norm(mu_x - mu_y, p=2)
    cosine    = 1 - F.cosine_similarity(mu_x.unsqueeze(0), mu_y.unsqueeze(0))
    return {"euclidean": euclidean.item(), "cosine": cosine.item()}
