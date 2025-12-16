from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

Array = np.ndarray


def alter_array(X: Array) -> Array:
    """Change the type of the array and make the shape (n_samples, n_features)"""
    # Changes the array to float64
    X = np.asarray(X, dtype=np.float64)
    # Shape is (n_samples, n_features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def rng(seed: Optional[int]) -> np.random.Generator:
    """Get a NumPy Generator from an int seed."""
    return np.random.default_rng(seed)


def euclidean_distance(X: Array, C: Array) -> Array:
    """Computes the squared Euclidean distances between X and C"""
    # Gettuing Row squared norms of X
    X_n = np.sum(X * X, axis=1, keepdims=True)
    # Gettuing Row squared norms of C
    C_n = np.sum(C * C, axis=1, keepdims=True).T
    # Matrix of dot products -> shape (n, k)
    XC = X @ C.T
    # Formula for pairwise squared distances
    return X_n - 2.0 * XC + C_n


def kmeans_init(X: Array, k: int, rng: np.random.Generator) -> Array:
    """k-means initialization, returns centers"""
    # Some intilization
    n, d = X.shape
    centers = np.empty((k, d), dtype=np.float64)

    # Choose first center randomly
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]

    # Distances from each point to ita nearest chosen center
    closest_sq_dists = euclidean_distance(X, centers[0:1]).ravel()

    # Performing the sampling
    for i in range(1, k):
        # Sample next center
        probs = closest_sq_dists.copy()
        # if all distances are identical, use uniform instead
        if probs.sum() <= 0:
            probs = np.ones_like(probs) / probs.size
        else:
            probs /= probs.sum()
        # Adding the data to the centers to the matrix
        idx = rng.choice(n, p=probs)
        centers[i] = X[idx]

        # Recompute each point's distance to the nearest center after center is added
        new_dists = euclidean_distance(X, centers[i:i+1]).ravel()
        closest_sq_dists = np.minimum(closest_sq_dists, new_dists)

    return centers

@dataclass
class KMeans:
    k: int
    max_iter: int = 300
    tol: float = 1e-5
    seed: Optional[int] = None

    # Learned attributes from .fit
    cluster_centers_: Array | None = None
    labels_: Array | None = None
    inertia_: float | None = None
    n_iterations: int | None = None

    def fit(self, X: Array) -> "KMeans":
        """Run K-Means algorithm on the data and store labels/centers/inertia"""
        # Prepare the array
        X = alter_array(X)
        # Initilize attributes
        n, d = X.shape
        gen = rng(self.seed)

        # Initialize centers with k-means
        centers = kmeans_init(X, self.k, gen)
        labels = np.empty(n, dtype=np.int64)
        inertia = np.inf

        # Do iterations of clustering
        for it in range(1, self.max_iter + 1):
            # Compute squared distances from each point to each center
            dists = euclidean_distance(X, centers)  # (n, k)
            # Assign each sample to the nearest center
            new_labels = np.argmin(dists, axis=1)

            # We start updating
            new_centers = centers.copy()
            for j in range(self.k):
                mask = (new_labels == j)
                if not np.any(mask):
                    # If a cluster becomes empty, reseed it
                    new_centers[j] = X[gen.integers(0, n)]
                else:
                    # Move center to the mean of its assigned points
                    new_centers[j] = X[mask].mean(axis=0)

            # Compute new sum of min squared distances
            new_dists = euclidean_distance(X, new_centers)
            new_inertia = float(np.sum(np.min(new_dists, axis=1)))

            # Checking convergence by measuring how much the centers moved
            center_shift = float(np.linalg.norm(new_centers - centers))

            # Update the state of the loop
            centers, labels, inertia = new_centers, new_labels, new_inertia

            # Stop if centers moved less than the given tolerance
            if center_shift <= self.tol:
                self.n_iter_ = it
                break
        else:
            # Loop exhausted without breaking -> reached max_iter
            self.n_iter_ = self.max_iter

        # Persist learned parameters on self
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X: Array) -> Array:
        """Assign each sample in X to the nearest learned center"""
        # Initilize the array as we need
        X = alter_array(X)
        # Getting the distance of clusters
        dists = euclidean_distance(X, self.cluster_centers_)
        # Returh the index with the minimum value
        return np.argmin(dists, axis=1)


@dataclass
class GMM:
    k: int
    max_iter: int = 200
    tol: float = 1e-4
    seed: Optional[int] = None
    cov_reg: float = 1e-6

    # Learned attributes after .fit is called
    means_: Array | None = None
    covariances_: Array | None = None
    weights_: Array | None = None
    responsibilities_: Array | None = None
    log_likelihood_history_: list[float] | None = None
    converged_: bool | None = None
    n_iter_: int | None = None

    def init_params(self, X: Array) -> Tuple[Array, Array, Array]:
        """Initialize (means, covariances, weights) for EM"""
        n, d = X.shape
        gen = rng(self.seed)

        # Start means at different random locations using k-means
        means = kmeans_init(X, self.k, gen)

        # Use the overall covariance of the dataset for each component
        X_center = X - X.mean(axis=0, keepdims=True)
        cov_global = (X_center.T @ X_center) / max(n - 1, 1)
        cov_global += self.cov_reg * np.eye(d)
        covariances = np.repeat(cov_global[None, :, :], self.k, axis=0)

        # Start by mixing weights
        weights = np.full(self.k, 1.0 / self.k, dtype=np.float64)
        return means, covariances, weights

    @staticmethod
    def logsumexp(Z: Array, axis: int = -1) -> Array:
        """Performing log(sum(exp(Z))"""
        m = np.max(Z, axis=axis, keepdims=True)
        return (m + np.log(np.sum(np.exp(Z - m), axis=axis, keepdims=True))).squeeze(axis)

    def log_density(self, X: Array, mean: Array, cov: Array) -> Array:
        """Log multivariate normal density for all rows in X"""
        d = X.shape[1]
        # Preping the covariance
        cov = cov + self.cov_reg * np.eye(d)
        try:
            # slogdet gives sign + log|cov|
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                # Handling edge cases that need regularization
                cov = cov + 10.0 * self.cov_reg * np.eye(d)
                sign, logdet = np.linalg.slogdet(cov)
            # Compute quadratic form
            diff = X - mean
            # Solving for cov * y = diff^T gives y
            sol = np.linalg.solve(cov, diff.T)
            quad = np.sum(diff.T * sol, axis=0)
            # Handling errors
        except np.linalg.LinAlgError:
            # Fallback for pathological cases: use pseudo-inverse
            cov_inv = np.linalg.pinv(cov)
            diff = X - mean
            quad = np.sum(diff * (diff @ cov_inv), axis=1)
            logdet = np.log(np.linalg.det(cov) + 1e-12)

        # Equating the constant using the d * log(2π) formula
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

    def fit(self, X: Array) -> "GMM":
        """Fit a full-covariance Gaussian Mixture with EM. Stops when improvement in total log-likelihood <= provided tolerance """
        # Initilizaing the array for fitting and associated parameters
        X = alter_array(X)
        n, d = X.shape
        means, covs, weights = self.init_params(X)
        ll_history: list[float] = []
        converged = False
        # Beginning iteration fpr both Expectation and Maximization steps
        for it in range(1, self.max_iter + 1):
            # Begin expectation step, calculate responsibility (belongingness to normal curve)
            log_resp = np.empty((n, self.k), dtype=np.float64)
            for k in range(self.k):
                log_resp[:, k] = np.log(weights[k] + 1e-16) + self.log_density(X, means[k], covs[k])

            # Normalize in log-space to avoid errors with flow
            log_norm = self.logsumexp(log_resp, axis=1)
            # responsibilities = exp(log_resp - log_norm)
            resp = np.exp(log_resp - log_norm[:, None])

            # Total log-likelihood is the sum over log p(x_i)
            ll = float(np.sum(log_norm))
            ll_history.append(ll)

            # Begin maximization step
            # Number of points in each cluster
            Nk = resp.sum(axis=0) + 1e-16
            weights = Nk / n

            # Get the Means
            means = (resp.T @ X) / Nk[:, None]

            # Getting the full covariances
            covs_new = np.empty((self.k, d, d), dtype=np.float64)
            for k in range(self.k):
                diff = X - means[k]                       # (n,d)
                # Weight each row of diff by r_{ik} and accumulate outer products
                cov_k = (diff * resp[:, [k]]).T @ diff / Nk[k]
                # Regularize to keep Σ_k positive-definite
                cov_k += self.cov_reg * np.eye(d)
                covs_new[k] = cov_k
            covs = covs_new

            # Check if we have converged given our initilized tolerance
            if it > 1:
                if abs(ll_history[-1] - ll_history[-2]) <= self.tol:
                    converged = True
                    self.n_iter_ = it
                    break
        else:
            # We have reached the maximum number of iterations without satisfying tolerance
            self.n_iter_ = self.max_iter

        # Re-populate with learned parameters
        self.means_ = means
        self.covariances_ = covs
        self.weights_ = weights
        self.responsibilities_ = resp
        self.log_likelihood_history_ = ll_history
        self.converged_ = converged
        return self

    def predict(self, X: Array) -> Array:
        """Return the most likely component index for each sample in X"""
        # Prepare matrix
        X = alter_array(X)
        n = X.shape[0]
        log_resp = np.empty((n, self.k), dtype=np.float64)
        # Iterate over samples and calculate its responsibility
        for k in range(self.k):
            log_resp[:, k] = np.log(self.weights_[k] + 1e-16) + self.log_density(X, self.means_[k], self.covariances_[k])
        # Argmax over components gives us the label assignments
        return np.argmax(log_resp, axis=1)

