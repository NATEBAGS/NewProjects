import pandas as pd
import numpy as np
import argparse

from decision_tree import tune_tree_depth, evaluate_sklearn_tree
from naives_bayes import NaiveBayesClassifier
from naives_bayes import prep_sklearn, evaluate_sklearn_nb
def load_mushroom(path: str):
    """Load the Mushroom dataset and return the dataframe of features (X) and the series of class labels (y)"""
    # Initilizing the column names with their respective attribute names that were provided in the names file
    colnames = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat"
    ]
    df = pd.read_csv(path, header=None, names=colnames)
    y = df["class"].astype(str)
    X = df.drop(columns=["class"]).astype(str)
    # Return our modified dataframe
    return X, y

def load_votes(path: str, missing_info: str = "?"):
    """Load the Congressional Voting Records dataset"""
    colnames = [
        "class",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa",
    ]

    # Read as raw strings; keep '?' literal
    df = pd.read_csv(path,header=None, names=colnames,na_filter=False,keep_default_na=False, skipinitialspace=True)

    # Making cells are strings
    df = df.map(lambda v: str(v).strip())
    # Separating the classes as democrat and republican
    y = df["class"]
    X = df.drop(columns=["class"])
    return X, y


def load_dataset(dataset_name: str):
    """Handles the experiment based on the dataset"""
    if dataset_name == "mushroom":
        # If the input is mushroom, the experiment will be done on the mushroom dataset
        return load_mushroom("mushroom/agaricus-lepiota.data")
    elif dataset_name == "votes":
        # If the dataset is votes, then the experiment will be done on the voting dataset
        return load_votes("congressional+voting+records/house-votes-84.data")


def split_80_10_10(X, y, train_frac=0.8, dev_frac=0.1, seed=10):
    """Split the dataset indices into train/dev/test subsets"""

    # Get total samples
    samples = len(y)

    # Create a way to index samples
    index = np.arange(samples)

    # Shuffling indices
    shuff = np.random.default_rng(seed=seed)
    shuff.shuffle(index)

    # Compute split sizes for train and dev sets
    total_train = int(train_frac * samples)
    total_dev = int(dev_frac * samples)

    # Slicing the indices into our train/dev/test splits
    train_index = index[:total_train]
    dev_index = index[total_train:total_train + total_dev]
    test_index = index[total_train + total_dev:]

    # Selecting rows by position from our features and labels
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_dev,   y_dev   = X.iloc[dev_index],   y.iloc[dev_index]
    X_test,  y_test  = X.iloc[test_index],  y.iloc[test_index]

    # Return our train/dev/test split
    return X_train, y_train, X_dev, y_dev, X_test, y_test


def evaluate_model(model, X_train, y_train, X_dev, y_dev, X_test, y_test):
    """Given a trained model and its splits compute accuracy for each split. Also, print confusion matrix of the data"""

    # Computing accuracy on each split using the score() helper
    train_acc = model.score(X_train, y_train)
    dev_acc   = model.score(X_dev, y_dev)
    test_acc  = model.score(X_test, y_test)

    # Outputting the results
    print("Model Accuracy:")
    print(f"train: {train_acc:.4f}")
    print(f"dev:   {dev_acc:.4f}")
    print(f"test:  {test_acc:.4f}")
    print()

    # Get confusion matrix on the test set using our helper function
    cm = model.confusion_matrix(X_test, y_test)

    print("Confusion Matrix:")
    # Printing it in a class order
    classes = sorted(set(y_test.astype(str).unique()) | set(model.classes_))
    # Labeling the confusion matrix
    header = "pred\\actual\t" + "\t".join(classes)
    print(header)
    for pred_label in classes:
        row_counts = [str(cm[pred_label][actual_label]) for actual_label in classes]
        print(f"{pred_label}\t\t" + "\t".join(row_counts))
    print()

    # Return metrics
    return {"train_acc": train_acc, "dev_acc": dev_acc, "test_acc": test_acc, "confusion_matrix": cm,}


def tuner(X_train, y_train, X_dev, y_dev, values):
    """Simple hyperparameter tuning and pick the one that scores best on the dev set"""

    # Setting up our variables for storing the best tuning features
    scores = {}
    best_model = None
    best_smooth = None
    best_acc = -1.0

    for smooth in values:
        # Intilialize a new model
        bayes = NaiveBayesClassifier(smoothing=smooth)

        # Fit it on the training data
        bayes.fit(X_train, y_train)

        # Get dev accuracy
        acc = bayes.score(X_dev, y_dev)

        # Store the score for later
        scores[smooth] = acc

        # Track the best-performing parameters for the model
        if acc > best_acc:
            best_acc = acc
            best_smooth = smooth
            best_model = bayes
    # Return the model with the best parameters for the job
    return best_model, best_smooth, scores

def run_naive_bayes(dataset: str):
    """Runs the Naive Bayes experiment for the chosen dataset"""

    # Load
    X, y = load_dataset(dataset)

    # Split
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_80_10_10(X, y, train_frac=0.8, dev_frac=0.1, seed=42)

    # Hyperparameter tuning for smoothing
    candidate_laplace_values = [1e-7, 0.1, 0.5, 1.0, 2.0, 5.0]
    best_model, best_smooth, smoothing_scores = tuner(X_train, y_train, X_dev, y_dev, candidate_laplace_values)

    # Outputing the tuning data
    print("Smoothing tuning (dev accuracy)")
    for lap, acc in smoothing_scores.items():
        print(f"Smoothing = {lap}: dev_acc={acc:.4f}")
    print(f"best smoothing based on dev: {best_smooth}")
    print()

    # Evaluate the Naive Bayes model on the split data
    print("Our Naive Bayes results:")
    our_results = evaluate_model(best_model, X_train, y_train, X_dev, y_dev, X_test, y_test)

    # Prepare data for sklearn CategoricalNB
    X_train_enc, X_dev_enc, X_test_enc, enc = prep_sklearn(X_train, X_dev, X_test)

    # Train and evaluate sklearn CategoricalNB
    print("sklearn CategoricalNB results:")
    sk_results = evaluate_sklearn_nb(X_train_enc, y_train, X_dev_enc, y_dev, X_test_enc, y_test)

    return {"ours": our_results, "sklearn": sk_results, "best_laplace": best_smooth, "laplace_scores": smoothing_scores,}

def run_decision_tree(dataset: str):
    """Runs the Naive Bayes experiment for the chosen dataset"""

    # Load
    X, y = load_dataset(dataset)

    # Split
    X_train, y_train, X_dev, y_dev, X_test, y_test = split_80_10_10(X, y, train_frac=0.8, dev_frac=0.1, seed=42)

    # Hyperparameter tuning for our decision tree
    candidate_depths = [1, 2, 3, 4, 5, 6, 8, 10]
    best_tree, best_depth, depth_scores = tune_tree_depth(X_train, y_train, X_dev, y_dev, candidate_depths)

    # Printing the tuning results
    print("Decision Tree max_depth tuning")
    for d, acc in depth_scores.items():
        print(f"max_depth={d}: dev_acc={acc:.4f}")
    print(f"best max_depth based on dev: {best_depth}")
    print()

    # Evaluate our tree on train/dev/test
    print("Our Decision Tree results:")
    our_tree_results = evaluate_model(best_tree, X_train, y_train,X_dev, y_dev, X_test, y_test)

    # Evaluate sklearn on same split
    print("sklearn DecisionTreeClassifier results:")
    sk_tree_results = evaluate_sklearn_tree(X_train, y_train, X_dev, y_dev, X_test, y_test, max_depth=best_depth)

    print("sklearn DecisionTreeClassifier Accuracy")
    print(f"train: {sk_tree_results['train_acc']:.4f}")
    print(f"dev:   {sk_tree_results['dev_acc']:.4f}")
    print(f"test:  {sk_tree_results['test_acc']:.4f}")
    print()

    print("sklearn DecisionTreeClassifier Confusion Matrix")
    cm = sk_tree_results["confusion_matrix"]
    classes = sorted(list(set(y_test.astype(str).unique())))
    header = "pred\\actual\t" + "\t".join(classes)
    print(header)
    for pred_label in classes:
        row_counts = [str(cm[pred_label][actual_label]) for actual_label in classes]
        print(f"{pred_label}\t\t" + "\t".join(row_counts))
    print()

    return {"ours_tree": our_tree_results, "sklearn_tree": sk_tree_results, "best_depth": best_depth, "depth_scores": depth_scores}


def main():
    # Function to run the code in the command line
    parser = argparse.ArgumentParser(description="Run Naive Bayes or Decision Tree on mushroom or voting dataset.")
    parser.add_argument("--dataset", type=str, choices=["mushroom", "votes"], required=True, help="Which dataset to use")
    parser.add_argument("--model", type=str, choices=["nb", "tree"], required=True, help="Which model pipeline to run: nb (Naive Bayes) or tree (Decision Tree)")
    args = parser.parse_args()
    # The choices for the ML algorithms
    if args.model == "nb":
        run_naive_bayes(args.dataset)
    elif args.model == "tree":
        run_decision_tree(args.dataset)
    else:
        raise ValueError("N/A")



if __name__ == "__main__":
    main()


