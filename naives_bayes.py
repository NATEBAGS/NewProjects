import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from math import log

class NaiveBayesClassifier:

    def __init__(self, smoothing=1.0):
        """Initilizing our paramaters for the experiment"""

        # Smoothing makes sure no 0 ends up on the denominator
        self.smoothing = smoothing

        # Setting params to be learned
        self.classes_ = None
        self.class_log_prior_ = {}
        self.feature_cond_log_prob_ = {}
        self.feature_value_vocab_ = {}

    def convert(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of X with NaN filled and everything cast to string."""
        # Copying the dataframe for making changes
        Xc = X.copy()
        # Iterating through the columns
        for col in Xc.columns:
            # Casting the columns to be strings
            Xc[col] = Xc[col].astype(str)
        # Returning our modified DataFrame that is ready for data processing
        return Xc

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Naive Bayes Estimation: P(feature=value | class)"""
        # Cleaning our Dataframe before fitting
        X = self.convert(X)
        # Casting our label numpy array so it can handle string formatting
        y = y.astype(str)
        # Getting the number of samples for later use
        n_samples = len(y)

        # Initlizing our counts to be held in dictionaries like {class: count}
        class_counts = y.value_counts().to_dict()

        # Storing the classes as lists
        self.classes_ = list(class_counts.keys())

        # For each class we compute P(class = c).
        for c in self.classes_:
            self.class_log_prior_[c] = log(class_counts[c] / n_samples)

        # Make sure case fit() is called once
        self.feature_cond_log_prob_.clear()
        self.feature_value_vocab_.clear()

        # Loop over each feature/column in X to estimate P(feature=value | class)
        for feat in X.columns:
            # Create an extra dict to hold this features conditional probability
            self.feature_cond_log_prob_[feat] = {}

            # Get the set of all possible values this feature has
            vocab = set(X[feat].unique().tolist())
            self.feature_value_vocab_[feat] = vocab

            # Setting the number of distinct possible values for this feature
            V = len(vocab)

            # Computing conditional probabilities for each class
            for c in self.classes_:
                # Extract the column values of this feature and count it
                vals_c = X[feat][y == c]

                # Getting occurrences of each value in this class.
                counts = vals_c.value_counts().to_dict()

                # Getting the number of rows of this class for this specific feature
                total_c = sum(counts.values())

                # initilizing a dictionary for this combination of feature and class
                self.feature_cond_log_prob_[feat][c] = {}

                # Iterate through every value in the feature
                for v in vocab:
                    # Getting the number of training examples in class c who's feature equals v
                    count_v = counts.get(v, 0)
                    # Getting the numerator for the calculation
                    num = count_v + self.smoothing
                    # Getting the denominator for the calculation
                    den = total_c + self.smoothing * V
                    # Calculating P(v | class)
                    self.feature_cond_log_prob_[feat][c][v] = log(num / den)

        # Return self to run the NaiveBayesClassifier on it
        return self


    def predict_one(self, row: pd.Series):
        """Returns the most likely class label for this one example"""
        # Initilize a best variable and lowest value so any score is better
        best_c = None
        best_score = -np.inf
        # Compute log P(class | row) for each possible class
        for c in self.classes_:
            # Setting the log prior
            logp = self.class_log_prior_[c]

            # We do log P(feature=value | class=c) for each feature in this row
            for feat, val in row.items():
                cond_table = self.feature_cond_log_prob_[feat][c]
                # Incrementing if we have seen it
                if val in cond_table:
                    logp += cond_table[val]

            # Find out if this class has the highest log score so far
            if logp > best_score:
                # Storing our best values
                best_score = logp
                best_c = c

        # Return the highest log-probability
        return best_c

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Clean the dataframe so formatting matches
        X = self.convert(X)

        # Set a list to collect predictions for each row
        predictions = []

        # Iterate over each row in our dataframe
        for _, row in X.iterrows():
            # Predict one label for that row and add it to the list
            predictions.append(self.predict_one(row))

        # Return predictions as a numpy array
        return np.array(predictions)


    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        # Get predicted labels on the features
        y_pred = self.predict(X)

        # Convert labels to string to numpy for later comparison
        y_true = y.astype(str).to_numpy()

        # Computing the mean to get our accuracy
        return np.mean(y_pred == y_true)


    def confusion_matrix(self, X: pd.DataFrame, y: pd.Series):
        """The confusion matrix for the decision tree (slightly different from naive bayes)"""
        # Predictions labels
        y_pred = self.predict(X)

        # Convert labels to string and numpy
        y_true = y.astype(str).to_numpy()

        # Get the all the classes we been seen or predicted
        classes = sorted(list(set(self.classes_) | set(y_true)))

        # Initialize the dictionary to default
        cm = {yp: {yt: 0 for yt in classes} for yp in classes}

        # Walk through each prediction for the cm data
        for yp, yt in zip(y_pred, y_true):
            cm[yp][yt] += 1

        # Return the confusion matrix
        return cm

def prep_sklearn(X_train, X_dev, X_test):
    """Makes sure the data is in the correct format for sklearn to handle"""

    # Making a copy of each split and casting it to strings
    Xt = X_train.copy().astype(str)
    Xd = X_dev.copy().astype(str)
    Xs = X_test.copy().astype(str)

    # Create and fit encoder for training set
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(Xt)

    # Transform splits to numpy arrays
    Xt_enc = enc.transform(Xt)
    Xd_enc = enc.transform(Xd)
    Xs_enc = enc.transform(Xs)

    return Xt_enc, Xd_enc, Xs_enc, enc

def evaluate_sklearn_nb(X_train_enc, y_train,X_dev_enc,   y_dev, X_test_enc,  y_test):
    """Train sklearn.naive_bayes.CategoricalNB and print statistics on the test set"""
    # Initialize sklearn's categorical NB
    sk_nb = CategoricalNB()

    # Fit on training data
    sk_nb.fit(X_train_enc, y_train.astype(str))

    # Predict splits
    y_prediction_train = sk_nb.predict(X_train_enc)
    y_prediction_dev = sk_nb.predict(X_dev_enc)
    y_prediction_test = sk_nb.predict(X_test_enc)

    # Compute accuracies
    train_accuracy = np.mean(y_prediction_train == y_train.astype(str).to_numpy())
    dev_accuracy = np.mean(y_prediction_dev   == y_dev.astype(str).to_numpy())
    test_accuracy = np.mean(y_prediction_test  == y_test.astype(str).to_numpy())

    # Build confusion matrix
    classes = sorted(list(set(y_test.astype(str).unique()) | set(sk_nb.classes_)))
    cm = {pred: {actual: 0 for actual in classes} for pred in classes}
    for yp, yt in zip(y_prediction_test, y_test.astype(str).to_numpy()):
        cm[yp][yt] += 1

    # Print results
    print("sklearn CategoricalNB Accuracy")
    print(f"train: {train_accuracy:.4f}")
    print(f"dev:   {dev_accuracy:.4f}")
    print(f"test:  {test_accuracy:.4f}")
    print()

    print("sklearn CategoricalNB Confusion Matrix")
    header = "prediction\\actual\t" + "\t".join(classes)
    print(header)
    for pred_label in classes:
        row_counts = [str(cm[pred_label][actual_label]) for actual_label in classes]
        print(f"{pred_label}\t\t" + "\t".join(row_counts))
    print()

    return {
        "train_acc": train_accuracy,
        "dev_acc": dev_accuracy,
        "test_acc": test_accuracy,
        "confusion_matrix": cm,
    }