import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

from models import KMeans, GMM
from helpers import (load_dataset, standardize, destandardize, ensure_dir, inertia as km_inertia, total_log_likelihood, gmm_num_parameters, aic as AIC, bic as BIC, silhouette_score, plot_clusters, plot_elbow, plot_aic_bic, save_data_json)

def destandardize_centers(centers_std: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Return centers in original units if training used standardization"""
    return destandardize(centers_std, mu, sigma)


def destandardize_params(means_std: np.ndarray, covs_std: np.ndarray, weights: np.ndarray,mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change GMM parameters from standardized units back to original units"""
    # Change each component mean back to the original feature unit
    means = means_std * sigma + mu
    # Build diagonal scaling matrix
    S = np.diag(sigma)
    # Create an output array with the same shape as covs_std
    covs = np.empty_like(covs_std)
    # Applying multivariate changing rule (for variables)
    for i in range(covs_std.shape[0]):
        covs[i] = S @ covs_std[i] @ S
    return means, covs, weights


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with two possible commands, run and whatclusters"""
    parser = argparse.ArgumentParser(
        description="Run KMeans/GMM implementations and save results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Adding the arguments for the run command on the data
    p_run = sub.add_parser("run", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_run.add_argument("--algo", choices=["kmeans", "gmm"], required=True)
    p_run.add_argument("--k", type=int, required=True)
    p_run.add_argument("--data", type=str, default="data/hw4_dataset.csv")
    p_run.add_argument("--outdir", type=str, default="outputs")
    p_run.add_argument("--standardize", action="store_true")
    p_run.add_argument("--seed", type=int, default=20)
    p_run.add_argument("--max-iter", type=int, default=300)
    p_run.add_argument("--tol", type=float, default=1e-4)
    p_run.add_argument("--cov-reg", type=float, default=1e-6)
    p_run.add_argument("--silhouette-sample", type=int, default=0)
    p_run.add_argument("--title", type=str, default=None)

    # Adding the arguments for the whatclusters command
    p_sw = sub.add_parser("whatclusters", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p_sw.add_argument("--algo", choices=["kmeans", "gmm"], required=True)
    p_sw.add_argument("--kmin", type=int, required=True)
    p_sw.add_argument("--kmax", type=int, required=True)
    p_sw.add_argument("--data", type=str, default="data/hw4_dataset.csv")
    p_sw.add_argument("--outdir", type=str, default="outputs")
    p_sw.add_argument("--standardize", action="store_true")
    p_sw.add_argument("--seed", type=int, default=20)
    p_sw.add_argument("--max-iter", type=int, default=300)
    p_sw.add_argument("--tol", type=float, default=1e-4)
    p_sw.add_argument("--cov-reg", type=float, default=1e-6)
    p_sw.add_argument("--silhouette-sample", type=int, default=0)

    return parser

def cmd_run(args: argparse.Namespace) -> None:
    """Run KMeans or GMM and output its cluster plot + data """
    # Load dataset
    X_orig = load_dataset(args.data)

    # Standardizee if needed
    if args.standardize:
        X, mu, sigma = standardize(X_orig)
    else:
        X, mu, sigma = X_orig.copy(), None, None

    # Prepare output directories and file paths
    outdir = Path(args.outdir)
    figdir = ensure_dir(outdir / "figures")
    resdir = ensure_dir(outdir / "results")
    tag = f"{args.algo}"
    fig_path = figdir / f"{tag}.png"
    params_path = resdir / f"{tag}_outputs.json"

    # Seed
    seed = int(args.seed)

    # Handle the case if kmeans is the input
    if args.algo == "kmeans":
        # Build and fit KMeans
        km = KMeans(k=args.k, max_iter=args.max_iter, tol=args.tol, seed=seed)
        km.fit(X)

        # If standardization happened, convert centers back to original units
        if args.standardize:
            centers = destandardize_centers(km.cluster_centers_, mu, sigma)
        else:
            centers = km.cluster_centers_

        # Predict labels on original data
        labels = km.predict(X)
        # Output the sizes of the clusters
        counts = np.bincount(labels, minlength=args.k)
        print(f"Cluster sizes (kmeans): {counts.tolist()}")

        # Get metrics on original units
        inertia_on_input = km_inertia(X_orig, labels, centers)
        sample = args.silhouette_sample if args.silhouette_sample > 0 else None
        sil = silhouette_score(X_orig, labels, sample_size=sample, seed=seed)

        # 6a) Plot clusters (first two dimensions) and save labels/params
        title = args.title or f"KMeans (k={args.k})"
        plot_clusters(X_orig, labels, centers=centers, title=title, out_path=fig_path)
        save_data_json(params_path, algo="kmeans", k=args.k, seed=seed, max_iter=args.max_iter, tol=float(args.tol), standardized=bool(args.standardize), cluster_centers=centers, inertia_on_input=float(inertia_on_input), silhouette_on_input=float(sil), n_iter=int(km.n_iter_ or args.max_iter))
        print(f"KMeans: Output = {params_path}\n Figure = {fig_path}")
    # Hndling gmm as other input
    else:
        # 2b) Build and fit GMM
        gm = GMM(k=args.k, max_iter=args.max_iter, tol=args.tol, seed=seed, cov_reg=args.cov_reg)
        gm.fit(X)

        # Put parameters back to original units if standardized
        if args.standardize:
            means, covs, weights = destandardize_params(gm.means_, gm.covariances_, gm.weights_, mu, sigma)
        else:
            means, covs, weights = gm.means_, gm.covariances_, gm.weights_

        # Getting our labels and outputting cluster sizes and mixture weights
        labels = gm.predict(X)
        counts = np.bincount(labels, minlength=args.k)
        print(f"Cluster sizes: {counts.tolist()}")
        print(f"Mixture weights: {weights.tolist()}")

        # Getting Metrics
        ll_total = total_log_likelihood(X_orig, means, covs, weights)
        d = X_orig.shape[1]
        p = gmm_num_parameters(args.k, d)
        aic_val = AIC(ll_total, p)
        bic_val = BIC(ll_total, p, X_orig.shape[0])
        sample = args.silhouette_sample if args.silhouette_sample > 0 else None
        sil = silhouette_score(X_orig, labels, sample_size=sample, seed=seed)

        # Plot clusters nd save it
        title = args.title or f"GMM (k={args.k})"
        plot_clusters(X_orig, labels, centers=means, title=title, out_path=fig_path)
        save_data_json(params_path, algo="gmm", k=args.k, seed=seed, max_iter=args.max_iter, tol=float(args.tol), cov_reg=float(args.cov_reg), standardized=bool(args.standardize), means=means, covariances=covs, weights=weights, log_likelihood_total_on_input=float(ll_total),
            aic=float(aic_val), bic=float(bic_val), silhouette_on_input=float(sil), converged=bool(gm.converged_), n_iter=int(gm.n_iter_ or args.max_iter))
        print(f"GMM data found at {params_path}\nFigure found at {fig_path}")

# Handling the other command "whatclusters"
def cmd_whatclusters(args: argparse.Namespace) -> None:
    """Generate elbow plot for KMeans or AIC/BIC plot for GMM"""
    # Load data
    X_orig = load_dataset(args.data)

    # Standardize for training
    if args.standardize:
        X, mu, sigma = standardize(X_orig)
    else:
        X, mu, sigma = X_orig.copy(), None, None

    outdir = Path(args.outdir)
    figdir = ensure_dir(outdir / "figures")

    ks = list(range(args.kmin, args.kmax + 1))

    if args.algo == "kmeans":
        # Collect inertia/silouhette
        inertias: List[float] = []
        sils: List[float] = []

        # Iterating through the different ks
        for k in ks:
            # Intilizing Kmeans and fitting the model
            km = KMeans(k=k, max_iter=args.max_iter, tol=args.tol, seed=args.seed)
            km.fit(X)
            if args.standardize:
                centers = destandardize_centers(km.cluster_centers_, mu, sigma)
            else:
                centers = km.cluster_centers_
            # Predict the labels
            labels = km.predict(X)
            # Add inertia to the list
            inertias.append(km_inertia(X_orig, labels, centers))
            # And Silhouetes
            sample = args.silhouette_sample if args.silhouette_sample > 0 else None
            sil = silhouette_score(X_orig, labels, sample_size=sample, seed=args.seed)
            sils.append(float(sil))

        # Save elbow plot
        elbow_path = figdir / f"kmeans_elbow.png"
        plot_elbow(ks, inertias, out_path=elbow_path, ylabel="Inertia (lower is better)")
        print(f"KMeans elbow plot: {elbow_path}")

        # If user gave a size, silhouette was computed
        if args.silhouette_sample > 0:
            sil_path = figdir / f"kmeans_silhouette.png"
            plot_elbow(ks, sils, out_path=sil_path, ylabel="Silhouette (higher is better)")
            print(f"KMeans silhouette plot: {sil_path}")

    else:
        # Prepping the data to be kept
        aics: List[float] = []
        bics: List[float] = []
        sils: List[float] = []

        # Iterating through for each k
        for k in ks:
            # Initilize the model and fit it to the data
            gm = GMM(k=k, max_iter=args.max_iter, tol=args.tol, seed=args.seed, cov_reg=args.cov_reg)
            gm.fit(X)
            if args.standardize:
                means, covs, weights = destandardize_params(gm.means_, gm.covariances_, gm.weights_, mu, sigma)
            else:
                means, covs, weights = gm.means_, gm.covariances_, gm.weights_
            # Compute selection criteria
            ll_total = total_log_likelihood(X_orig, means, covs, weights)
            p = gmm_num_parameters(k, X_orig.shape[1])
            aics.append(AIC(ll_total, p))
            bics.append(BIC(ll_total, p, X_orig.shape[0]))
            labels = gm.predict(X)
            sample = args.silhouette_sample if args.silhouette_sample > 0 else None
            sil = silhouette_score(X_orig, labels, sample_size=sample, seed=args.seed)
            sils.append(float(sil))

        # Save AIC/BIC plot
        ab_path = figdir / f"gmm_aic/bic_plot.png"
        plot_aic_bic(ks, aics, bics, out_path=ab_path)
        print(f"GMM AIC/BIC plot: {ab_path}")

        # If silhouette parameter was given then save plot as well
        if args.silhouette_sample > 0:
            sil_path = figdir / f"gmm_silhouette_plot.png"
            plot_elbow(ks, sils, out_path=sil_path, ylabel="Silhouette (higher is better)")
            print(f"GMM silhouette plot: {sil_path}")


# Function so the code can be ran through the command line
def main(argv: Optional[list[str]] = None) -> None:
    # Initilize the command line parser
    parser = build_parser()
    args = parser.parse_args(argv)
    # If the user wants to run the code
    if args.cmd == "run":
        cmd_run(args)
    # If the user wants to run the code and get elbow plot
    elif args.cmd == "whatclusters":
        cmd_whatclusters(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
