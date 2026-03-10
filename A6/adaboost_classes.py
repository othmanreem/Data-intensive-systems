#!/usr/bin/env python3
"""
Helper module to import AdaBoost classes without running module-level code.

This module re-exports the AdaBoostEnsemble and WeightedDecisionTree classes
from classification_adaboost.py, but without triggering the module-level
data loading and training code.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from typing import List


class WeightedDecisionTree(DecisionTreeClassifier):
    """
    A wrapper around DecisionTreeClassifier that properly handles sample weights.
    This tree is grown based on weighted training errors.
    """
    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, random_state: int = 42):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(self, X, y, sample_weight=None):
        """Fit the decision tree with optional sample weights."""
        return super().fit(X, y, sample_weight=sample_weight)


class AdaBoostEnsemble(BaseEstimator, ClassifierMixin):
    """
    AdaBoost ensemble of decision trees where each tree is grown based on
    weighted training errors. Weights are updated based on the error of
    previous trees.

    The algorithm:
    1. Initialize equal weights for all training samples
    2. For each tree in the ensemble:
       - Train a decision tree on weighted data
       - Calculate weighted error rate
       - Compute tree weight (alpha)
       - Update sample weights (increase for misclassified, decrease for correct)
       - Normalize weights
    3. Make predictions using weighted voting
    """

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees: List[WeightedDecisionTree] = []
        self.tree_weights: List[float] = []
        self.n_classes: int = 0
        self.classes_: np.ndarray = None

    def _initialize_weights(self, n_samples: int) -> np.ndarray:
        """Initialize equal weights for all samples."""
        return np.ones(n_samples) / n_samples

    def _update_weights(
        self,
        weights: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Update sample weights based on prediction errors.
        Increase weight for misclassified samples, decrease for correct.
        """
        # Misclassified samples get multiplied by exp(alpha)
        # Correctly classified samples get multiplied by exp(-alpha)
        misclassified = y_true != y_pred
        updated_weights = weights * np.exp(alpha * misclassified.astype(float))

        # Normalize weights
        return updated_weights / updated_weights.sum()

    def _compute_weighted_error(
        self,
        weights: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Compute weighted error rate."""
        misclassified = (y_true != y_pred).astype(float)
        return np.sum(weights * misclassified) / np.sum(weights)

    def _compute_alpha(self, error: float) -> float:
        """
        Compute the weight of the classifier.
        Avoid division by zero and log(0).
        """
        if error <= 0:
            return 10.0  # Very high weight for perfect classifier
        if error >= 1:
            return -10.0  # Very negative weight for completely wrong classifier
        return 0.5 * np.log((1 - error) / error)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostEnsemble':
        """Fit the AdaBoost ensemble."""
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)

        # Initialize sample weights
        weights = self._initialize_weights(n_samples)

        for i in range(self.n_estimators):
            # Create and train decision tree with current weights
            tree = WeightedDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            tree.fit(X, y, sample_weight=weights)

            # Make predictions
            y_pred = tree.predict(X)

            # Calculate weighted error
            error = self._compute_weighted_error(weights, y, y_pred)

            # Compute tree weight (alpha)
            alpha = self._compute_alpha(error)

            # Update sample weights
            weights = self._update_weights(weights, y, y_pred, alpha)

            # Store tree and its weight
            self.trees.append(tree)
            self.tree_weights.append(alpha)

            print(f"Tree {i+1}/{self.n_estimators}: Error={error:.4f}, Alpha={alpha:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted voting."""
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Get class labels
        classes = self.classes_

        # Compute weighted votes for each class
        n_samples = X.shape[0]
        weighted_votes = np.zeros((n_samples, len(classes)))

        for tree_idx, tree in enumerate(self.trees):
            alpha = self.tree_weights[tree_idx]
            predictions = all_predictions[tree_idx]

            for class_idx, class_label in enumerate(classes):
                weighted_votes[:, class_idx] += alpha * (predictions == class_label)

        # Return class with highest weighted vote
        return classes[np.argmax(weighted_votes, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using weighted voting."""
        # Get predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Get class labels
        classes = self.classes_

        # Compute weighted vote proportions for each class
        n_samples = X.shape[0]
        weighted_votes = np.zeros((n_samples, len(classes)))

        total_weight = sum(abs(w) for w in self.tree_weights)

        for tree_idx, tree in enumerate(self.trees):
            alpha = self.tree_weights[tree_idx]
            predictions = all_predictions[tree_idx]

            for class_idx, class_label in enumerate(classes):
                weighted_votes[:, class_idx] += abs(alpha) * (predictions == class_label)

        # Normalize to get probabilities
        return weighted_votes / total_weight
