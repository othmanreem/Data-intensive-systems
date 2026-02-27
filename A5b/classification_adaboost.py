import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')
np.random.seed(42)

REPO_ROOT    = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR     = os.path.join(REPO_ROOT, 'Datasets_all')
OUT_DIR      = Path('models')
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5
CHAMPION_F1  = 0.6110   # Score from A4


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


def evaluate_cv(model, X, y, cv, name='Model'):
    """Evaluate model using cross-validation."""
    scoring = {
        'accuracy' : 'accuracy',
        'f1'       : 'f1_weighted',
        'precision': 'precision_weighted',
        'recall'   : 'recall_weighted',
    }
    cv_res = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        'Model'         : name,
        'Accuracy_mean' : cv_res['test_accuracy'].mean(),
        'Accuracy_std'  : cv_res['test_accuracy'].std(),
        'F1_mean'       : cv_res['test_f1'].mean(),
        'F1_std'        : cv_res['test_f1'].std(),
        'Precision_mean': cv_res['test_precision'].mean(),
        'Recall_mean'   : cv_res['test_recall'].mean(),
        '_f1_scores'    : cv_res['test_f1'],
    }


# Load data
movement_features_df = pd.read_csv(os.path.join(DATA_DIR, 'aimoscores.csv'))
weaklink_scores_df   = pd.read_csv(os.path.join(DATA_DIR, 'scores_and_weaklink.csv'))

print('Movement features shape:', movement_features_df.shape)
print('Weak link scores shape:', weaklink_scores_df.shape)

DUPLICATE_NASM_COLS = [
    'No_1_NASM_Deviation',
    'No_2_NASM_Deviation',
    'No_3_NASM_Deviation',
    'No_4_NASM_Deviation',
    'No_5_NASM_Deviation',
]

movement_features_df = movement_features_df.drop(columns=DUPLICATE_NASM_COLS)
print('Shape after duplicate removal:', movement_features_df.shape)

weaklink_categories = [
    'ExcessiveForwardLean', 'ForwardHead', 'LeftArmFallForward',
    'LeftAsymmetricalWeightShift', 'LeftHeelRises', 'LeftKneeMovesInward',
    'LeftKneeMovesOutward', 'LeftShoulderElevation', 'RightArmFallForward',
    'RightAsymmetricalWeightShift', 'RightHeelRises', 'RightKneeMovesInward',
    'RightKneeMovesOutward', 'RightShoulderElevation',
]

weaklink_scores_df['WeakestLink'] = (
    weaklink_scores_df[weaklink_categories].idxmax(axis=1)
)
print('Weakest Link class distribution:')
print(weaklink_scores_df['WeakestLink'].value_counts())

# Merge Datasets
target_df = weaklink_scores_df[['ID', 'WeakestLink']].copy()
merged_df = movement_features_df.merge(target_df, on='ID', how='inner')
print('Merged dataset shape:', merged_df.shape)

EXCLUDE_COLS    = ['ID', 'WeakestLink', 'EstimatedScore']
feature_columns = [c for c in merged_df.columns if c not in EXCLUDE_COLS]

X = merged_df[feature_columns].values
y = merged_df['WeakestLink'].values

print(f'Feature matrix shape : {X.shape}')
print(f'Number of features   : {len(feature_columns)}')
print(f'Number of classes    : {len(np.unique(y))}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f'Training samples : {X_train.shape[0]}')
print(f'Test samples     : {X_test.shape[0]}')

cv_strategy = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

# Train AdaBoost ensemble
print("\n" + "="*60)
print("TRAINING ADABOOST ENSEMBLE")
print("="*60)

adaboost_model = AdaBoostEnsemble(
    n_estimators=50,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE
)

adaboost_model.fit(X_train_scaled, y_train)

# Cross-validation
adaboost_cv = evaluate_cv(
    adaboost_model, X_train_scaled, y_train, cv_strategy,
    name='AdaBoost Ensemble'
)

# Test set evaluation
adaboost_model.fit(X_train_scaled, y_train)
y_pred_adaboost = adaboost_model.predict(X_test_scaled)

test_f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')
test_acc_adaboost = accuracy_score(y_test, y_pred_adaboost)
test_prec_adaboost = precision_score(y_test, y_pred_adaboost, average='weighted', zero_division=0)
test_rec_adaboost = recall_score(y_test, y_pred_adaboost, average='weighted', zero_division=0)

print("\n" + "="*60)
print("ADABOOST RESULTS")
print("="*60)
print(f'CV F1: {adaboost_cv["F1_mean"]:.4f} +/- {adaboost_cv["F1_std"]:.4f}')
print(f'Test F1: {test_f1_adaboost:.4f}')
print(f'Test Accuracy: {test_acc_adaboost:.4f}')
print(f'Test Precision: {test_prec_adaboost:.4f}')
print(f'Test Recall: {test_rec_adaboost:.4f}')

# Compare with baseline models
rf_champion = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)

rf_cv = evaluate_cv(
    rf_champion, X_train_scaled, y_train, cv_strategy,
    name='Random Forest (Baseline)'
)

rf_champion.fit(X_train_scaled, y_train)
y_pred_rf = rf_champion.predict(X_test_scaled)
test_f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print("\n" + "="*60)
print("COMPARISON WITH BASELINE")
print("="*60)
print(f'Random Forest CV F1: {rf_cv["F1_mean"]:.4f} +/- {rf_cv["F1_std"]:.4f}')
print(f'Random Forest Test F1: {test_f1_rf:.4f}')

# Statistical significance test
def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    k        = len(scores_a)
    diff     = scores_a - scores_b
    d_bar    = diff.mean()
    s_sq     = diff.var(ddof=1)
    var_corr = (1/k + n_test/n_train) * s_sq
    t_stat   = d_bar / np.sqrt(var_corr)
    p_value  = 2 * (1 - stats.t.cdf(abs(t_stat), df=k-1))
    return float(t_stat), float(p_value)

n_total      = len(X_train_scaled)
n_test_fold  = n_total // N_SPLITS
n_train_fold = n_total - n_test_fold

result_map   = {
    'AdaBoost Ensemble': adaboost_cv['_f1_scores'],
    'Random Forest': rf_cv['_f1_scores']
}

adaboost_scores = result_map['AdaBoost Ensemble']
rf_scores = result_map['Random Forest']

t, p = corrected_resampled_ttest(adaboost_scores, rf_scores, n_train_fold, n_test_fold)
print(f"\nStatistical Test (AdaBoost vs Random Forest):")
print(f"  t-statistic: {t:+.3f}")
print(f"  p-value: {p:.4f}")
print(f"  Significant at α=0.05: {'Yes' if p < 0.05 else 'No'}")

# Save model
artifact = {
    'model'                  : adaboost_model,
    'model_name'             : 'AdaBoost Ensemble',
    'scaler'                 : scaler,
    'feature_columns'        : feature_columns,
    'cv_metrics': {
        'f1_mean'      : float(adaboost_cv['F1_mean']),
        'f1_std'       : float(adaboost_cv['F1_std']),
        'accuracy_mean': float(adaboost_cv['Accuracy_mean']),
    },
    'test_metrics': {
        'f1'       : float(test_f1_adaboost),
        'accuracy' : float(test_acc_adaboost),
        'precision': float(test_prec_adaboost),
        'recall'   : float(test_rec_adaboost),
    },
    'a4_champion_f1' : CHAMPION_F1,
    'improvement_pct': float((test_f1_adaboost - CHAMPION_F1) / CHAMPION_F1 * 100),
}

out_path = OUT_DIR / 'adaboost_classification.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(artifact, f)

print(f'\nSaved model to: {out_path}')

# Classification report
print('\nCLASSIFICATION REPORT: AdaBoost Ensemble')
print(classification_report(y_test, y_pred_adaboost, zero_division=0))

# Feature importance analysis (simplified)
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Calculate feature importance as average across all trees
all_importances = np.zeros(len(feature_columns))
for tree in adaboost_model.trees:
    all_importances += tree.feature_importances_

avg_importances = all_importances / len(adaboost_model.trees)
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': avg_importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['Importance'].values)
plt.yticks(range(len(top_features)), top_features['Feature'].values)
plt.xlabel('Average Feature Importance')
plt.ylabel('Features')
plt.title('Top 15 Feature Importance - AdaBoost Ensemble')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / 'adaboost_feature_importance.png', dpi=150)
plt.close()

print(f"\nSaved feature importance plot to: {OUT_DIR / 'adaboost_feature_importance.png'}")
