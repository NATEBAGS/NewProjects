from typing import Iterable, Optional, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import euclidean_distance

Array = np.ndarray

def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist"""
    dir = Path(path)
    dir.mkdir(parents=True, exist_ok=True)
    return dir


def load_dataset(path: str | Path, usecols: Optional[Iterable[str]] = None) -> Array:
    """Load a CSV into a float64 NumPy array for later use"""
    # Read the CSV
    df = pd.read_csv(path, usecols=usecols)
    # Convert to a float64 matrix
    X = df.to_numpy(dtype=np.float64)
    # Handling edge cases
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def standardize(X: Array) -> Tuple[Array, Array, Array]:
    """Z-score standardization on columns"""
    # Make sure array has correct type
    X = np.asarray(X, dtype=np.float64)
    # Get the column means
    mu = X.mean(axis=0)
    # Get the standard deviation
    sigma = X.std(axis=0, ddof=0)
    sigma_safe = np.where(sigma == 0.0, 1.0, sigma)
    X_std = (X - mu) / sigma_safe
    return X_std, mu, sigma_safe


def destandardize(X_std: Array, mu: Array, sigma: Array) -> Array:
    """Undo standardization if needed """
    return X_std * sigma + mu


def inertia(X: Array, labels: Array, centers: Array) -> float:
    """For KMeans, sum of squared distances of samples to their center"""
    # Initilize array attributes
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.float64)
    # Compute all distances and pick distance to each sample's center
    D2 = euclidean_distance(X, centers)
    return float(np.sum(D2[np.arange(X.shape[0]), labels]))


def logsumexp(Z: Array, axis: int = -1) -> Array:
    """Performing log(sum(exp(Z))"""
    m = np.max(Z, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(Z - m), axis=axis, keepdims=True))).squeeze(axis)


def log_density(X: Array, mean: Array, cov: Array, cov_reg: float = 1e-6) -> Array:
    """Same functionality as one in models.py, just a different signature"""
    d = X.shape[1]
    cov = cov + cov_reg * np.eye(d)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 10.0 * cov_reg * np.eye(d)
            sign, logdet = np.linalg.slogdet(cov)
        diff = X - mean
        sol = np.linalg.solve(cov, diff.T)
        quad = np.sum(diff.T * sol, axis=0)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
        diff = X - mean
        quad = np.sum(diff * (diff @ cov_inv), axis=1)
        logdet = np.log(np.linalg.det(cov) + 1e-12)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

def total_log_likelihood(X: Array, means: Array, covs: Array, weights: Array, cov_reg: float = 1e-6) -> float:
    """Total log-likelihood"""
    # Initlize the array
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = means.shape[0]
    log_resp = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        log_resp[:, j] = np.log(weights[j] + 1e-16) + log_density(X, means[j], covs[j], cov_reg=cov_reg)
    # Sum over components
    return float(np.sum(logsumexp(log_resp, axis=1)))


def gmm_num_parameters(k: int, d: int) -> int:
    """Number of free parameters for GMM"""
    cov_params = d * (d + 1) // 2
    return k * (d + cov_params) + (k - 1)


def aic(log_likelihood: float, num_params: int) -> float:
    """Akaike Information Criterion: AIC = 2p - 2L"""
    return 2.0 * num_params - 2.0 * log_likelihood


def bic(log_likelihood: float, num_params: int, n_samples: int) -> float:
    """Bayesian Information Criterion: BIC = p log n - 2L"""
    return num_params * np.log(max(n_samples, 1)) - 2.0 * log_likelihood

def euclidean_distances(A: Array, B: Array) -> Array:
    """Pairwise Euclidean distances"""
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True)
    AB = A @ B.T
    D2 = np.maximum(A2 - 2.0 * AB + B2, 0.0)
    return np.sqrt(D2, dtype=np.float64)


def silhouette_score(X: Array, labels: Array, sample_size: Optional[int] = None, seed: Optional[int] = None) -> float:
    """Compute mean silhouette scores"""
    # Setup arrays with expected data types
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = X.shape[0]

    # Subsample for speed (large n)
    if sample_size is not None and 0 < sample_size < n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_size, replace=False)
        X = X[idx]
        labels = labels[idx]
        n = X.shape[0]

    # Handle the case where there's only one cluster or all clusters are singles
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2 or np.all(counts == 1):
        return float("nan")

    # Compute all pairwise distances
    D = euclidean_distances(X, X)

    # Inter cluster mean distance
    a = np.zeros(n, dtype=np.float64)
    # Nearest other mean distance cluster
    b = np.full(n, np.inf, dtype=np.float64)

    # Compute a(i) and b(i) for every cluster
    for c in unique:
        mask_c = (labels == c)
        idx_c = np.where(mask_c)[0]

        if idx_c.size == 1:
            # Single cluster (no neighbors)
            a[idx_c] = 0.0
        else:
            # Get within-cluster distances
            Dc = D[np.ix_(idx_c, idx_c)]
            a[idx_c] = (Dc.sum(axis=1) - np.diag(Dc)) / (idx_c.size - 1)

        # b(i)
        for c2 in unique:
            if c2 == c:
                continue
            mask_c2 = (labels == c2)
            idx_c2 = np.where(mask_c2)[0]
            if idx_c2.size == 0:
                continue
            # Cross-cluster distances
            D_cross = D[np.ix_(idx_c, idx_c2)]
            # For each i in c, get mean distance to cluster c2
            means = D_cross.mean(axis=1)
            # Keep the smallest mean
            b[idx_c] = np.minimum(b[idx_c], means)

    # avoid division by 0
    denom = np.maximum(a, b)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(denom > 0, (b - a) / denom, 0.0)
    # Return the mean silhouette (ignore NaNs if any)
    return float(np.nanmean(s))


def save(fig: plt.Figure, out_path: str | Path) -> None:
    """Save a matplotlib Figure"""
    out_path = Path(out_path)
    # Create path if needed
    ensure_dir(out_path.parent)
    # Save the created figure
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_clusters(X: Array, labels: Array, centers: Optional[Array] = None, title: str | None = None, out_path: Optional[str | Path] = None) -> Optional[Path]:
    """Scatter plot of points (indicating clusters)"""
    # Make sure the array is of the correct type
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)  # <-- make sure labels are ints
    d = X.shape[1]

    # Choose the axes
    x0 = X[:, 0]
    x1 = X[:, 1] if d > 1 else np.zeros_like(x0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Draw data colored by cluster
    ax.scatter(x0, x1, c=labels, s=20, alpha=0.85)

    # overlay centers
    if centers is not None:
        centers = np.asarray(centers, dtype=np.float64)
        c0 = centers[:, 0]
        c1 = centers[:, 1] if centers.shape[1] > 1 else np.zeros_like(c0)
        ax.scatter(c0, c1, marker="X", s=120, edgecolor="k", linewidths=1.0)

    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    if title:
        ax.set_title(title)
    # Save the figure
    if out_path is None:
        plt.show()
        return None
    save(fig, out_path)
    return Path(out_path)



def plot_elbow(ks: Iterable[int], scores: Iterable[float], out_path: str | Path, ylabel: str = "Inertia") -> Path:
    """Make an elbow-style plot for K-Means"""
    ks = list(ks)
    scores = list(scores)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, scores, marker="o")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel(ylabel)
    ax.set_title("Model selection curve")
    save(fig, out_path)
    return Path(out_path)


def plot_aic_bic(ks: Iterable[int], aics: Iterable[float], bics: Iterable[float], out_path: str | Path) -> Path:
    """Plot AIC and BIC for GMM model selection"""
    ks = list(ks)
    aics = list(aics)
    bics = list(bics)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, aics, marker="o", label="AIC")
    ax.plot(ks, bics, marker="o", label="BIC")
    ax.set_xlabel("k (number of components)")
    ax.set_ylabel("Criterion value (lower is better)")
    ax.set_title("GMM model selection")
    ax.legend()
    save(fig, out_path)
    return Path(out_path)

def save_data_json(path: str | Path, **kwargs) -> Path:
    """Save model dta as JSON."""
    def _to_native(v):
        # Convert NumPy types recursively
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {k: _to_native(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_native(x) for x in v]
        return v
    # Convert fields
    blob = {k: _to_native(v) for k, v in kwargs.items()}
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        # Print it nicely in the json
        json.dump(blob, f, indent=2)
    return path
