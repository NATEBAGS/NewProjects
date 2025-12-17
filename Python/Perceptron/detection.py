import argparse, re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Importing my perceptron class
from models.perceptron import Perceptron
from experiments.spambase import tune_logreg, tune_linsvc
from experiments.spambase import eval_with_stats, standardize_fit_transform
# Creating a list that includes the English alphabet in order
ALPHABET = [chr(c) for c in range(ord('a'), ord('z')+1)]
# Initilizing some standard cues that occur in English and Dutch
COMMON_CUES= ["th","he","er","de","en","aa","ij","oo","ee","sch"]

def read_lines(path: Path):
    """Function that will read all the lines from a file"""
    # Open the file
    with open(path, "r", encoding="utf-8") as f:
        # We need to take out blank lines and whitespace
        lines = [ln.strip() for ln in f if ln.strip()]
    # We return a list of strings as sentences
    return lines

def letter_counts(s: str):
    """This function gets the letter counts of a string"""
    # We will make the string lowercase for consistent comparisons
    s = s.lower()
    # We will only count the characters from a-z
    count = Counter(ch for ch in s if 'a' <= ch <= 'z')
    # Return a vector that converted the list of integer into floats that can be easily used by sklearn models
    return np.array([count.get(a, 0) for a in ALPHABET], dtype=float)

def extra_features(s: str):
    """Defining a function for extra features when detecting English and Dutch words, Common cues, vowel ratio + average length."""
    # Convert the string to lowercase
    s_low = s.lower()
    # Creating a list for the feature
    feat = []
    # Loop through some of the defined COMMON_CUES
    for cue in COMMON_CUES:
        # Adding a count for the common cues
        feat.append(s_low.count(cue))
    # Handling vowel ratio by getting the number of vowels in the string
    vowels = sum(1 for ch in s_low if ch in "aeiou")
    # The total number of letters in the string
    letters = sum(1 for ch in s_low if 'a' <= ch <= 'z')
    # Add it to the list and if no letters were in the string return 0
    feat.append((vowels / letters) if letters else 0.0)

    # Getting the average tokens using regex library
    tokens = re.findall(r"[A-Za-z]+", s)
    # Finding the average token length
    feat.append((sum(map(len, tokens)) / len(tokens)) if tokens else 0.0)
    # Return an array with the features added to it
    return np.array(feat, dtype=float)

def featurize(sentences, add_feats=True):
    """It converts a list of sentences into a feature matrix that is ready for classifiers"""
    # Turns each sentence into a 26 dimension vector of letter counts
    X_letters = np.vstack([letter_counts(s) for s in sentences])
    # If add_feats is true, then it will add the extra features built for this problem
    if add_feats:
        # Stacking the features I created
        X_extra = np.vstack([extra_features(s) for s in sentences])
        # Adding the columns together for the created features
        return np.hstack([X_letters, X_extra])
    # If the extra features were not included, you just get a matrix with the counts of the letters
    return X_letters

def build_split(train_en, train_du, dev_en, dev_du, test_en, test_du, use_extras=True):
    """Building the split for the train/dev/test sets"""
    # Initilizing the training set using the provided text, labels: 1=English, 0=Dutch
    X_tr = featurize(train_en + train_du, add_feats=use_extras)
    y_tr = np.array([1]*len(train_en) + [0]*len(train_du))

    # Dev features
    X_dev = featurize(dev_en + dev_du, add_feats=use_extras)
    # Dev labels
    y_dev = np.array([1]*len(dev_en) + [0]*len(dev_du))

    # Test features
    X_te  = featurize(test_en + test_du, add_feats=use_extras)
    # Test labels
    y_te  = np.array([1]*len(test_en) + [0]*len(test_du))
    # Returns all splits as (features, labels)
    return X_tr, y_tr, X_dev, y_dev, X_te, y_te


def main():
    # Initilizing the parser
    p = argparse.ArgumentParser()
    # Initilizing our train/dev/test sets respectively
    p.add_argument("--train_en", default="data/english.txt")
    p.add_argument("--train_du", default="data/dutch.txt")
    p.add_argument("--dev_en",   default="data/dev_english.txt")
    p.add_argument("--dev_du",   default="data/dev_dutch.txt")
    p.add_argument("--test_en",  default="data/test_english.txt")
    p.add_argument("--test_du",  default="data/test_dutch.txt")
    p.add_argument("--no_extras", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Loading our train/dev/test sentences
    tr_en = read_lines(Path(args.train_en))
    tr_du = read_lines(Path(args.train_du))
    dv_en = read_lines(Path(args.dev_en))
    dv_du = read_lines(Path(args.dev_du))
    te_en = read_lines(Path(args.test_en))
    te_du = read_lines(Path(args.test_du))

    # Featurizing and building the vectors per split
    X_tr, y_tr, X_dev, y_dev, X_te, y_te = build_split(tr_en, tr_du, dv_en, dv_du, te_en, te_du, use_extras=not args.no_extras)
    # Fitting the scaler on TRAIN, and transforming all splits
    X_tr, X_dev, X_te = standardize_fit_transform(X_tr, X_dev, X_te)

    # Tuning the perceptron

    best = None
    # Setting our limits of epochs
    for it in [5,10,20,30, 50]:
        # Instantiating the perceptron on train
        clf = Perceptron(max_iter=it, learning_rate=1.0, shuffle=True, random_state=args.seed).fit(X_tr, y_tr)
        # Evaluate on the dev set
        acc, _ = clf.evaluate(X_dev, y_dev)
        if best is None or acc > best[0]:
            # Store the best epoch so far
            best = (acc, it, clf)
    best_acc, best_it, _ = best
    print("\nPerceptron Data")
    print(f"Best dev set accuracy = {best_acc:.4f} with max_iter = {best_it}")  # log choice
    # Fitting the data on train before evaluating on test
    clf_p = Perceptron(max_iter=best_it, learning_rate=1.0, shuffle=True, random_state=args.seed).fit(X_tr, y_tr)
    # Output the statistics for Perceptron on the test set
    eval_with_stats("Test Perceptron", clf_p, X_te, y_te, label_order=(0,1))

    # Printing statistics for the Linear Logistic Regression model
    print("Logistic Regression Data")
    # Finding the best C
    dev_acc, bestC = tune_logreg(X_tr, y_tr, X_dev, y_dev, seed=args.seed)
    # Outputting the best C
    print(f"Best dev set accuracy = {dev_acc:.4f} with C = {bestC}")
    # Train final Logistic Regression model
    logreg = LogisticRegression(C=bestC, penalty="l2", solver="liblinear", max_iter=2000, random_state=args.seed).fit(X_tr, y_tr)
    # Output the statistics on the test set
    eval_with_stats("Test Logistic Regression", logreg, X_te, y_te, label_order=(0,1))

    # Printing statistics for the Linear SVC model similar to all the code above, read above if necessary
    print("LinearSVC Data")
    dev_acc_svc, bestC_svc = tune_linsvc(X_tr, y_tr, X_dev, y_dev, seed=args.seed)
    print(f"Best dev set accuracy = {dev_acc_svc:.4f} with C = {bestC_svc}")
    lsvc = LinearSVC(C=bestC_svc, random_state=args.seed).fit(X_tr, y_tr)
    eval_with_stats("Test LinearSVC", lsvc, X_te, y_te, label_order=(0,1))

if __name__ == "__main__":
    main()
