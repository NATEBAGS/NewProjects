import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.cluster import KMeans as SKKMeans
from sklearn.mixture import GaussianMixture as SKGMM
from helpers import (load_dataset, standardize, destandardize, ensure_dir, inertia as km_inertia, silhouette_score, total_log_likelihood, gmm_num_parameters, aic as AIC, bic as BIC, plot_clusters, save_data_json)

def build_parser() -> argparse.ArgumentParser:
    """Create an argparse parser for command line"""
    p = argparse.ArgumentParser(description="Run scikit-learn baselines (KMeans/GMM) and save outputs.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--algo", choices=["kmeans", "gmm"], required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--data", type=str, default="data/hw4_dataset.csv")
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--seed", type=int, default=20)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--silhouette-sample", type=int, default=0)
    p.add_argument("--title", type=str, default=None)
    return p


def destandardize_kmeans(centers_std: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return centers in original units: centers = centers_std * sigma + mu."""
    return destandardize(centers_std, mu, sigma)


def _destandardize_gmm_params(means_std: np.ndarray, covs_std: np.ndarray, weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map GMM parameters from standardized space back to original units"""
    # De-standardize the means
    means = means_std * sigma + mu

    # Creating a scaling marix and using it
    d = means.shape[1]
    S = np.diag(sigma)  # (d,d)
    covs = np.empty_like(covs_std)
    for i in range(covs_std.shape[0]):
        covs[i] = S @ covs_std[i] @ S
    return means, covs, weights



def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load data
    X_orig = load_dataset(args.data)

    # Standardize the data if needed
    if args.standardize:
        X, mu, sigma = standardize(X_orig)
    else:
        X, mu, sigma = X_orig.copy(), None, None

    # Handle errors where directories don't exist
    outdir = Path(args.outdir)
    figdir = ensure_dir(outdir / "figures")
    resdir = ensure_dir(outdir / "results")

    # Commonly used strings/paths
    tag = f"{args.algo}_sklearn_k{args.k}"
    fig_path = figdir / f"{tag}.png"
    params_path = resdir / f"{tag}_outputs.json"

    # Fit the chosen sklearn model
    if args.algo == "kmeans":
        # Build sklearn KMeans
        sk = SKKMeans(n_clusters=args.k, n_init=10, max_iter=args.max_iter, random_state=args.seed)

        # Fit oon the data
        labels = sk.fit_predict(X)

        # Centers are returned in the training space
        centers_std = sk.cluster_centers_
        if args.standardize:
            centers = destandardize_kmeans(centers_std, mu, sigma)
        else:
            centers = centers_std

        # Compute inertia of the models
        km_inertia_orig = km_inertia(X_orig, labels, centers)

        # Get the silhouette score
        sample = args.silhouette_sample if args.silhouette_sample > 0 else None
        sil = silhouette_score(X_orig, labels, sample_size=sample, seed=args.seed)

        # Figure title
        title = args.title or f"sklearn KMeans (k={args.k})"

        # Plot the clusters
        plot_clusters(X_orig, labels, centers=centers, title=title, out_path=fig_path)

        # Save data to a json
        save_data_json(params_path, algo="kmeans", k=args.k, seed=args.seed, max_iter=args.max_iter, standardized=args.standardize, cluster_centers=centers, inertia_on_input=km_inertia_orig, silhouette_on_input=float(sil))

        # Print to console
        print(f"sklearn KMeans saved params -> {params_path}")
        print(f"Figure found at {fig_path}")

    # The other argument is GMM
    else:
        # Build sklearn Gaussian Mixture Model
        gm = SKGMM( n_components=args.k, covariance_type="full", max_iter=args.max_iter, random_state=args.seed, init_params="kmeans")

        # Fit on the data and get labels
        gm.fit(X)
        labels = gm.predict(X)

        # Parameters are in training space; map back to original for plotting/saving.
        means_std = gm.means_
        covs_std = gm.covariances_
        weights = gm.weights_
        if args.standardize:
            means, covs, weights = _destandardize_gmm_params(means_std, covs_std, weights, mu, sigma)
        else:
            means, covs = means_std, covs_std

        n = X.shape[0]
        # sklearn's score returns log-likelihood per sample
        ll_sklearn_total = float(gm.score(X) * n)

        # Use helpers for plotting
        ll_total = total_log_likelihood(X_orig, means, covs, weights)

        # Calculate BIC and AIC on original data
        d = X_orig.shape[1]
        p = gmm_num_parameters(args.k, d)
        aic_val = AIC(ll_total, p)
        bic_val = BIC(ll_total, p, n)

        # Output silhouette of data
        sample = args.silhouette_sample if args.silhouette_sample > 0 else None
        sil = silhouette_score(X_orig, labels, sample_size=sample, seed=args.seed)

        # Title
        title = args.title or f"sklearn GMM (k={args.k})"

        # Plot clusters
        plot_clusters(X_orig, labels, centers=means, title=title, out_path=fig_path)

        # Save data from the experiment
        save_data_json(params_path, algo="gmm", k=args.k, seed=args.seed, max_iter=args.max_iter, standardized=args.standardize, means=means, covariances=covs, weights=weights, log_likelihood_total_on_input=ll_total, log_likelihood_total_sklearn_space=ll_sklearn_total, aic=aic_val, bic=bic_val,
            silhouette_on_input=float(sil), converged=bool(gm.converged_), n_iter=int(gm.n_iter_))

        # Print to console to tell user where stuff is
        print(f"sklearn GMM saved params -> {params_path}")
        print(f"Figure found at {fig_path}")


if __name__ == "__main__":
    main()
