import numpy as np
import pandas as pd
from math import log2
from sklearn.tree import DecisionTreeClassifier

class TreeNode:
    """Defines what a node is in the decision tree"""

    # Intilizing things we need for a decision tree
    def __init__(self, is_leaf, predicted_class, feature=None, children=None):
        self.is_leaf = is_leaf
        self.predicted_class = predicted_class
        self.feature = feature
        self.children = children or {}

def entropy_of_labels(y_series: pd.Series) -> float:
    """Compute entropy using the formula from class"""
    # Counts the appearance of each class
    counts = y_series.value_counts()
    total = len(y_series)
    ent = 0.0
    # Iterate through the classes
    for c in counts.index:
        # Probability of class c
        p_c = counts[c] / total
        # Entropy term
        ent -= p_c * log2(p_c)
    return ent


def information_gain(X_col: pd.Series, y: pd.Series) -> float:
    """Computing information gain using the formula from class"""
    # Get the parent entropy
    base_entropy = entropy_of_labels(y)

    # Setting up some variables prior to iteration
    total = len(y)
    weighted_sum = 0.0

    # groupby gets us each distinct value in this column
    for v, idxs in X_col.groupby(X_col).groups.items():
        # Labels of only the rows where this feature has the value of v
        y_sub = y.loc[idxs]
        # The fraction of samples that have this value v
        weight = len(y_sub) / total
        # Find the entropy of the subset
        subset_entropy = entropy_of_labels(y_sub)
        # Add up the weighted entropy
        weighted_sum += weight * subset_entropy
    # Return our information gain
    return base_entropy - weighted_sum

class SimpleDecisionTree:
    """A basic decision tree that chooses splits based on the information gain"""

    def __init__(self, max_depth=5, min_samples_split=2):
        # Some initilizations for things we need to know for our decision tree
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_list_ = None
        self.classes_ = None

    def to_strings(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make sure all features are treated as strings"""
        # Make a copy of the dataframe
        Xc = X.copy()
        # Make sure every column has type string
        for col in Xc.columns:
            Xc[col] = Xc[col].astype(str)
        return Xc

    def fit_tree(self, X: pd.DataFrame, y: pd.Series):
        """Train the tree on the given training subset."""
        # Convert the frame to features and labels as strings
        Xc = self.to_strings(X)
        yc = y.astype(str)

        # Save the names for later use
        self.feature_list_ = list(Xc.columns)
        self.classes_ = sorted(list(set(yc.to_numpy())))

        # Build the tree
        self.root = self.build_node(Xc, yc, depth=0, available_features=self.feature_list_)

        return self

    def build_node(self, X, y, depth, available_features):
        """Recursively build nodes"""

        # Count how many labels we have for each class
        class_counts = y.value_counts().to_dict()
        # Get the majority class from the counts
        majority_class = max(class_counts, key=class_counts.get)

        # Stop if all the labels are the same
        if len(class_counts) == 1:
            return TreeNode(is_leaf=True, predicted_class=majority_class)

        # Stop if we hit the max depth of the tree
        if depth >= self.max_depth:
            return TreeNode(is_leaf=True, predicted_class=majority_class)

        # Stop if not enough samples
        if len(y) < self.min_samples_split:
            return TreeNode(is_leaf=True, predicted_class=majority_class)

        # Stop if there are no features left
        if len(available_features) == 0:
            return TreeNode(is_leaf=True, predicted_class=majority_class)

        # If no stoppage happens we choose best feature that is given by information gain
        best_feature = None
        # Set as lowest value so any score is better
        best_gain = -np.inf

        # Looping through to find the best information gain
        for feature in available_features:
            gain = information_gain(X[feature], y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature

        # If the best gain is less than 0 splitting wont help
        if best_gain <= 0:
            return TreeNode(is_leaf=True, predicted_class=majority_class)

        children = {}
        # Branching on every unique value that best_feature has
        values_for_feature = X[best_feature].unique().tolist()

        # We wont split below the node after splitting
        child_available = [f for f in available_features if f != best_feature]


        for v in values_for_feature:
            # Choose the rows where the best feature equals v
            selector = (X[best_feature] == v)
            # Get the rows with that value
            X_sub = X[selector]
            # Get the labels of those rows
            y_sub = y[selector]

            # Handling an edge case
            if len(y_sub) == 0:
                child_node = TreeNode(is_leaf=True, predicted_class=majority_class)
            else:
                # Go deeper
                child_node = self.build_node(X_sub, y_sub, depth + 1,child_available)
            # Store this subtreeas the value v
            children[v] = child_node

        # Get an internal node
        return TreeNode(is_leaf=False, predicted_class=majority_class, feature=best_feature,children=children)

    def predict_one_row(self, row: pd.Series, node: TreeNode):
        """Follow the tree down for a single row until we hit a leaf"""
        # Our base case is a leaf
        if node.is_leaf:
            return node.predicted_class

        # Get the rows value for the split feature
        feature_val = str(row[node.feature])

        # If we have a child for that value, recurse downwards
        if feature_val in node.children:
            return self.predict_one_row(row, node.children[feature_val])
        else:
            # Unseen value means we go back to node's majority class
            return node.predicted_class

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict a label for every row in X"""
        # Make sure everything are strings
        Xc = self.to_strings(X)
        predictions = []
        # Go through all the rows and add our predictions to alist
        for _, row in Xc.iterrows():
            predictions.append(self.predict_one_row(row, self.root))
        # Convert that list into a numpy array for later use
        return np.array(predictions)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """ Find the Accuracy on (X,y)."""
        # Get predictions
        y_pred = self.predict(X)
        # Cast the actual labels to an array
        y_string = y.astype(str).to_numpy()
        # Return the accuracy
        return np.mean(y_pred == y_string)

    def confusion_matrix(self, X: pd.DataFrame, y: pd.Series):
        """Confusion matrix for the tree"""
        # Predict the labels
        y_prediction = self.predict(X)
        # Make everything strings
        y_true = y.astype(str).to_numpy()

        # Get all da classes that are in y_true or predictions
        classes = sorted(list(set(y_true)))
        # Initilize the dict to default
        cm = {pred: {actual: 0 for actual in classes} for pred in classes}

        # Count how often our prediction was true
        for yp, yt in zip(y_prediction, y_true):
            cm[yp][yt] += 1

        return cm


def tune_tree_depth(X_train, y_train, X_dev, y_dev, selection_depths):
    """Hyperparameter tuning for the decision tree"""
    # Setting the variables needed to tune
    best_tree = None
    best_depth = None
    best_acc = -1.0
    depth_scores = {}

    for depth in selection_depths:
        # Make new tree with the depth of the iteration
        tree = SimpleDecisionTree(max_depth=depth, min_samples_split=2)
        # Fit the tree to the training data
        tree.fit_tree(X_train, y_train)

        # Evaluate the accuracy on the dev set
        acc = tree.score(X_dev, y_dev)
        # Keep the data
        depth_scores[depth] = acc

        # Keep the best tree
        if acc > best_acc:
            best_acc = acc
            best_depth = depth
            best_tree = tree

    return best_tree, best_depth, depth_scores


def evaluate_sklearn_tree(X_train, y_train, X_dev, y_dev, X_test, y_test, max_depth=None):
    """Train sklearn's DecisionTreeClassifier given the split"""

    # Convert data for consistency
    def to_string_df(D):
        Dc = D.copy()
        for c in Dc.columns:
            Dc[c] = Dc[c].astype(str)
        return Dc
    # Call the helper to convert the split into something sklearn can handle
    Xtr = to_string_df(X_train)
    Xdv = to_string_df(X_dev)
    Xte = to_string_df(X_test)

    # Building the mapping for every column
    encoders = {}
    for col in Xtr.columns:
        vals = sorted(Xtr[col].unique().tolist())
        mapping = {val: i for i, val in enumerate(vals)}
        encoders[col] = mapping

    # Function to convert dataframe into a matrix
    def encode_with_train_mapping(D):
        arr = np.zeros((len(D), len(D.columns)), dtype=int)
        for j, col in enumerate(D.columns):
            mapping = encoders[col]
            # unknown values -> -1
            arr[:, j] = [mapping.get(str(val), -1) for val in D[col]]
        return arr

    # Encode the sets that are split
    Xtr_enc = encode_with_train_mapping(Xtr)
    Xdv_enc = encode_with_train_mapping(Xdv)
    Xte_enc = encode_with_train_mapping(Xte)

    # Train sklearn tree
    sk_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=10)
    # Fit on training data
    sk_tree.fit(Xtr_enc, y_train.astype(str))

    # Predictions for each split
    y_predict_train = sk_tree.predict(Xtr_enc)
    y_predict_dev   = sk_tree.predict(Xdv_enc)
    y_predict_test  = sk_tree.predict(Xte_enc)

    # Convert the results to find accuracy
    y_train_true = y_train.astype(str).to_numpy()
    y_dev_true   = y_dev.astype(str).to_numpy()
    y_test_true  = y_test.astype(str).to_numpy()

    # Compute accuracy
    train_acc = np.mean(y_predict_train == y_train_true)
    dev_acc   = np.mean(y_predict_dev   == y_dev_true)
    test_acc  = np.mean(y_predict_test  == y_test_true)

    # Confusion matrix on test
    classes = sorted(list(set(y_test_true)))
    cm = {pred: {actual: 0 for actual in classes} for pred in classes}
    for yp, yt in zip(y_predict_test, y_test_true):
        cm[yp][yt] += 1

    return {"sklearn_tree": sk_tree, "train_acc": train_acc, "dev_acc": dev_acc, "test_acc": test_acc, "confusion_matrix": cm}