# This file consists of the perceptron algorithm implementation for this project
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

def toBin(y: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Map arbitrary binary labels to {-1, +1} and remember original (neg, pos)
    """
    # Turning to 1D numpy array
    y = np.asarray(y).ravel()
    # Using the unique labels for values
    uniq = np.unique(y)
    # Smaller value are negative, larger ones are positive
    neg_label, pos_label = uniq[0], uniq[1]
    # Larger is mapped to +1, smaller is mapped to -1
    y_pm1 = np.where(y == pos_label, 1, -1)
    # We return mapped labels and original ordering
    return y_pm1, (neg_label, pos_label)

def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """
    Return 2x2 confusion matrix with dimensions: [[TN, FP], [FN, TP]]
    """
    # Defining which values are going to be negative and positive
    neg, pos = labels
    # Making sure the 1D arrays are set up appropriately
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    # Calculating true negatives
    tn = np.sum((y_true == neg) & (y_pred == neg))
    # Calculating false positives
    fp = np.sum((y_true == neg) & (y_pred == pos))
    # # Calculating false negatives
    fn = np.sum((y_true == pos) & (y_pred == neg))
    # Calculating true positives
    tp = np.sum((y_true == pos) & (y_pred == pos))
    # Return a 2 x 2 integer matrix
    return np.array([[tn, fp], [fn, tp]], dtype=int)

@dataclass
class Perceptron:
    """
    Building mistake-driven Perceptron (bias + weights)
    """
    # How many times the training set is gone over
    max_iter: int = 10
    # Update rule
    learning_rate: float = 1.0
    # Helps Perceptron updates when shuffling happens and adds randomness (to improve accuracy)
    shuffle: bool = True
    random_state: Optional[int] = None
    # Setting the bias term
    fit_intercept: bool = True

    # Learned weights set by fit
    w_: Optional[np.ndarray] = None
    # Learned bias set by fit
    b_: float = 0.0
    # (neg_label, pos_labe) from the training data
    _label_order: Optional[Tuple[int, int]] = None  # (neg_label, pos_label)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train on (X, y) using the provided perceptron loop
        """
        # Standardizing as 2D array
        X = np.asarray(X)
        # Maping labels to {-1, +1}
        y_pm1, label_order = toBin(y)
        # Storing mapping predictions for later use
        self._label_order = label_order
        # Initilizing the dimensions for the dataset
        n_samples, n_features = X.shape
        # Using an instance that uses random_state
        rng = np.random.default_rng(self.random_state)

        # Initilizing the weights to 0's
        self.w_ = np.zeros(n_features, dtype=float)
        # Initilizing the bias to 0
        self.b_ = 0.0 if self.fit_intercept else 0.0  # kept for clarity

        # Begin the epoch loop
        for _ in range(self.max_iter):
            # Initilizing the count the udpates that happen during the loop
            updates = 0
            # shuffling the order for model traversal
            if self.shuffle:
                # Get a random permutation
                idx = rng.permutation(n_samples)
            else:
                # Getting the deterministic order
                idx = np.arange(n_samples)
            #We need to iterate over the examples
            for i in idx:
                # a = w dot x + b
                a = float(np.dot(self.w_, X[i]) + (self.b_ if self.fit_intercept else 0.0))
                # If something was misclassified or on boundary, we update (error-driven)
                if y_pm1[i] * a <= 0.0:
                    # Updating the weights
                    self.w_ += self.learning_rate * y_pm1[i] * X[i]
                    if self.fit_intercept:
                        # Bias gets updated if needed
                        self.b_ += self.learning_rate * y_pm1[i]
                    # Track updates when made
                    updates += 1
            # If no updates were made it converged, so we can exit the loop
            if updates == 0:
                break
        # We return self to be able to have chaining
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return a = X @ w + b"""
        # Ensure numpy array is used
        X = np.asarray(X)
        # Return our activations that are vectorized
        return X @ self.w_ + (self.b_ if self.fit_intercept else 0.0)

    def predict_bin(self, X: np.ndarray) -> np.ndarray:
        """Internal Predictions: sign(decision_func) """
        a = self.decision_function(X)
        # Setting the sign threshold at 0 (ties are +1)
        return np.where(a >= 0.0, 1, -1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using original label space (neg_label, pos_label).
        """
        # Unpacking original labels
        neg, pos = self._label_order
        # Internal predictions
        y_pm1 = self.predict_bin(X)
        # Mapping +1 to positive and -1 to negative
        return np.where(y_pm1 == 1, pos, neg)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy = mean(y_pred == y) in label space (original)"""
        # Predicted original labels
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y).ravel()))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convenience: will return (accuracy, confusion_matrix)
        """
        # Predicted labels in original space
        y_pred = self.predict(X)
        # Formula for accuracy
        acc = float(np.mean(y_pred == np.asarray(y).ravel()))
        # 2x2 matrix with original ordering (neg, pos)
        cm = confusion_matrix(y, y_pred, labels=self._label_order)  # type: ignore
        # Return a pair of accuracy and the confusion matrix
        return acc, cm