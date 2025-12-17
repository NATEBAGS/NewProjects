import argparse
import numpy as np
import pandas as pd

# Importing modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

# Importing Perceptron and confusion matrix functions
from models.perceptron import confusion_matrix
from models.perceptron import Perceptron

def load_spambase(path: str):
    """Assumes input is a CSV with numeric features and label in the last column."""
    # Read the csv file
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy()
    # Ensure labels are 0/1
    unique = np.unique(y)
    if not set(unique).issubset({0, 1}):
        # Map the smaller value to 0, larger to 1
        y = (y == unique.max()).astype(int)
    return X, y

def split_80_10_10(X, y, random_state=42):
    """Splitting our data into a test (10%), dev (10%), train (80%)"""
    # Creating the 10% test set data
    X_tr_dev, X_te, y_tr_dev, y_te = train_test_split(
        X, y, test_size=0.10, random_state=random_state, stratify=y
    )
    # Creating 10% dev set from the remaining 90% of data
    X_tr, X_dev, y_tr, y_dev = train_test_split(
        X_tr_dev, y_tr_dev, test_size=1/9, random_state=random_state, stratify=y_tr_dev
    )
    return X_tr, y_tr, X_dev, y_dev, X_te, y_te

def standardize_fit_transform(X_tr, X_dev, X_te):
    """Standardize features based on the train set."""
    # Setting up the z-score scaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    # Fitting on train only, then transform
    X_tr_s = scaler.fit_transform(X_tr)
    # Transform dev set using train's statistical data
    X_dev_s = scaler.transform(X_dev)
    # Transform test set using train's statistical data
    X_te_s  = scaler.transform(X_te)
    # Return the splits (scaled)
    return X_tr_s, X_dev_s, X_te_s

def evaluate(clf: Perceptron, X, y, name="set"):
    """This function creates the confusion matrix and outputs model accuracy"""
    # Evaluate accuracy and the confusion matrix
    acc, cm = clf.evaluate(X, y)
    # Print our accuracy and 2x2 confusion matrix
    print(f"{name} accuracy: {acc:.4f}")
    print(f"{name} confusion matrix:\n{cm}\n")
    return acc, cm

def eval_with_stats(name, clf, X, y, label_order=(0,1)):
    """Evaluate a trained classifier on (X, y) and prints accuracy/confusion matrix"""
    # Vector of predicted class labels
    y_pred = clf.predict(X)
    # Calculating accuracy
    acc = (y_pred == y).mean()
    # Outputs the 2x2 confusion matrix with given label ordering
    cm = confusion_matrix(y, y_pred, labels=label_order)

    # Computes precision/recall/F1 per class and returns arrays with metrics for each class
    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, labels=list(label_order), average=None
    )
    #index of the positive class in arrays above
    pos_idx = 1
    # Print accuracy, confusion matrix, and spam metrics
    print(f"{name} accuracy: {acc:.4f}")
    print(f"{name} confusion matrix:\n{cm}")
    print(f"precision={prec[pos_idx]:.4f}, recall={rec[pos_idx]:.4f}, f1={f1[pos_idx]:.4f}\n")

    # Return numbers for tables/logging
    return acc, cm, (prec[pos_idx], rec[pos_idx], f1[pos_idx])


def tune_logreg(X_tr, y_tr, X_dev, y_dev, seed=42, grid_C=(0.01, 0.1, 1.0, 10.0)):
    """Dev-set tuning for LogisticRegression"""
    # Holds (acc, C) of the best dev score
    best = None
    # Iterate for regularization
    for C in grid_C:
        # Instantiating the Linear Regression model for this C
        clf = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000, random_state=seed).fit(X_tr, y_tr)
        # Evaluating accuracy on dev set
        acc = clf.score(X_dev, y_dev)
        if (best is None) or (acc > best[0]):
            # Keeping track of ideal dev accuracy and C
            best = (acc, C)
    return best


def tune_linsvc(X_tr, y_tr, X_dev, y_dev, seed=42, grid_C=(0.01, 0.1, 1.0, 10.0)):
    """Dev-set tuning for LinearSVC"""
    # Holds the accuracy and C for the best dev set score
    best = None
    # Iterates over C candidates
    for C in grid_C:
        # Training the linearSVC on the train set only
        clf = LinearSVC(C=C, random_state=seed).fit(X_tr, y_tr)
        # Evaluates the accuracy on the dev set
        acc = clf.score(X_dev, y_dev)
        if (best is None) or (acc > best[0]):
            # Keep the best accuracy and C for the linearSVC
            best = (acc, C)
    return best



def main():
    """This main function will run the training of the models and output statistics for all models"""
    # Create a parser
    parser = argparse.ArgumentParser()
    # Adding all the necessary arguments that the model will need
    parser.add_argument("--data", default="/Users/natebagchee/PycharmProjects/142HW2/data/spambase.data", help="Path to spambase CSV")
    parser.add_argument("--max_iter_grid", nargs="+", type=int, default=[5, 10, 20, 30])
    parser.add_argument("--lr_grid", nargs="+", type=float, default=[1.0])  # classic perceptron uses 1.0
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling within epochs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Read csv = (feature matrix, label vector)
    X, y = load_spambase(args.data)
    # This is our 80-10-10 split
    X_tr, y_tr, X_dev, y_dev, X_te, y_te = split_80_10_10(X, y, random_state=args.seed)
    # Scaling using train data, then will apply to dev and test
    X_tr, X_dev, X_te = standardize_fit_transform(X_tr, X_dev, X_te)


    # Best will hold the best accuracy, best iteration and whatnot
    best = None
    # Loop over the epoch
    for max_iter in args.max_iter_grid:
        # Loop over the learning rates
        for lr in args.lr_grid:
            # Creating a perceptron with current parameters
            clf = Perceptron(max_iter=max_iter, learning_rate=lr,  shuffle=not args.no_shuffle, random_state=args.seed,).fit(X_tr, y_tr)
            # Score on developer set
            acc_dev, _ = clf.evaluate(X_dev, y_dev)
            key = (max_iter, lr)
            if (best is None) or (acc_dev > best[0]):
                # Store the best accuracy, parameters, and trained model
                best = (acc_dev, key, clf)

    best_acc, (best_iter, best_lr), best_clf = best
    print(f"Best dev: acc={best_acc:.4f} with max_iter={best_iter}, lr={best_lr}\n")

   # Merging train and dev features
    X_tr_full = np.vstack([X_tr, X_dev])
    # Merging labels
    y_tr_full = np.concatenate([y_tr, y_dev])
    # Create perceptrom with optimal parameters
    clf_final = Perceptron(max_iter=best_iter, learning_rate=best_lr, shuffle=not args.no_shuffle, random_state=args.seed,).fit(X_tr_full, y_tr_full)

    # Evaluates the accuracy of the perceptron on the test set
    print("\nPerceptron")
    evaluate(clf_final, X_te, y_te, name="Test perceptron")

    # Running the logistic regression (tuned by helper functions) over the test set
    print("\nLogistic Regression")
    best_acc_lr, best_C_lr = tune_logreg(X_tr, y_tr, X_dev, y_dev, seed=args.seed)
    print(f"Best dev set accuracy = {best_acc_lr:.4f} with C = {best_C_lr}")
    logreg_final = LogisticRegression(
        C=best_C_lr, penalty="l2", solver="liblinear", max_iter=2000, random_state=args.seed
    ).fit(np.vstack([X_tr, X_dev]), np.concatenate([y_tr, y_dev]))
    eval_with_stats("Test Logistic Regression", logreg_final, X_te, y_te, label_order=(0, 1))

    # Running the LinearSVC (tuned by helper functions) over the test set
    print("LinearSVC")
    best_acc_svc, best_C_svc = tune_linsvc(X_tr, y_tr, X_dev, y_dev, seed=args.seed)
    print(f"Best dev set accuracy = {best_acc_svc:.4f} with C = {best_C_svc}")
    linsvc_final = LinearSVC(C=best_C_svc, random_state=args.seed).fit(
        np.vstack([X_tr, X_dev]), np.concatenate([y_tr, y_dev])
    )
    eval_with_stats("Test LinearSVC", linsvc_final, X_te, y_te, label_order=(0, 1))


if __name__ == "__main__":
    main()
