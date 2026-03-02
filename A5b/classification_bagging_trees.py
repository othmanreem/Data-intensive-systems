import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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

# Encode string labels to integers for XGBoost/LightGBM compatibility
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cv_strategy = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

def evaluate_cv(model, X, y, cv, name='Model', use_encoded_labels=False):
    scoring = {
        'accuracy' : 'accuracy',
        'f1'       : 'f1_weighted',
        'precision': 'precision_weighted',
        'recall'   : 'recall_weighted',
    }
    y_to_use = y_encoded if use_encoded_labels else y
    cv_res = cross_validate(model, X, y_to_use, cv=cv, scoring=scoring)
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

# Baseline: Single Decision Tree
single_tree = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_STATE
)
single_tree_cv = evaluate_cv(
    single_tree, X_scaled, y, cv_strategy,
    name='Single Decision Tree'
)
print('SINGLE DECISION TREE')
print(f'CV F1: {single_tree_cv["F1_mean"]:.4f} +/- {single_tree_cv["F1_std"]:.4f}')

# Bagging with Decision Trees (default: uses all features)
bagging_default = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    n_estimators=200,
    max_samples=1.0,  # Bootstrap sample size (100% of training data)
    max_features=1.0,  # Use all features
    bootstrap=True,
    bootstrap_features=False,  # Don't subsample features
    n_jobs=-1,
    random_state=RANDOM_STATE
)
bagging_default_cv = evaluate_cv(
    bagging_default, X_scaled, y, cv_strategy,
    name='Bagging (All Features)'
)
print(f'Bagging (All Features) CV F1: {bagging_default_cv["F1_mean"]:.4f} +/- {bagging_default_cv["F1_std"]:.4f}')

# Bagging with Decision Trees + Feature Subsetting (Random Subspace Method)
# This creates trees using random subsets of predictors
bagging_subspace = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    n_estimators=200,
    max_samples=1.0,
    max_features=0.7,  # Use 70% of features for each tree
    bootstrap=True,
    bootstrap_features=True,  # Subsample features for each tree
    n_jobs=-1,
    random_state=RANDOM_STATE
)
bagging_subspace_cv = evaluate_cv(
    bagging_subspace, X_scaled, y, cv_strategy,
    name='Bagging (70% Features)'
)
print(f'Bagging (70% Features) CV F1: {bagging_subspace_cv["F1_mean"]:.4f} +/- {bagging_subspace_cv["F1_std"]:.4f}')

# Bagging with smaller feature subset (50%)
bagging_50features = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    n_estimators=200,
    max_samples=1.0,
    max_features=0.5,  # Use 50% of features for each tree
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
bagging_50features_cv = evaluate_cv(
    bagging_50features, X_scaled, y, cv_strategy,
    name='Bagging (50% Features)'
)
print(f'Bagging (50% Features) CV F1: {bagging_50features_cv["F1_mean"]:.4f} +/- {bagging_50features_cv["F1_std"]:.4f}')

# Bagging with even smaller feature subset (30%)
bagging_30features = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    n_estimators=200,
    max_samples=1.0,
    max_features=0.3,  # Use 30% of features for each tree
    bootstrap=True,
    bootstrap_features=True,
    n_jobs=-1,
    random_state=RANDOM_STATE
)
bagging_30features_cv = evaluate_cv(
    bagging_30features, X_scaled, y, cv_strategy,
    name='Bagging (30% Features)'
)
print(f'Bagging (30% Features) CV F1: {bagging_30features_cv["F1_mean"]:.4f} +/- {bagging_30features_cv["F1_std"]:.4f}')

# Compare with Random Forest (for reference)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # sqrt(n_features) - standard random forest
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_cv = evaluate_cv(
    rf_model, X_scaled, y, cv_strategy,
    name='Random Forest (sqrt features)'
)
print(f'Random Forest CV F1: {rf_cv["F1_mean"]:.4f} +/- {rf_cv["F1_std"]:.4f}')

# Compare with XGBoost and LightGBM (for reference)
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1,
    verbosity=0
)
xgb_cv = evaluate_cv(
    xgb_model, X_scaled, y, cv_strategy,
    name='XGBoost',
    use_encoded_labels=True
)
print(f'XGBoost CV F1: {xgb_cv["F1_mean"]:.4f} +/- {xgb_cv["F1_std"]:.4f}')

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    class_weight='balanced',
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)
lgb_cv = evaluate_cv(
    lgb_model, X_scaled, y, cv_strategy,
    name='LightGBM',
    use_encoded_labels=True
)
print(f'LightGBM CV F1: {lgb_cv["F1_mean"]:.4f} +/- {lgb_cv["F1_std"]:.4f}')

# Collect all results
all_results = [
    single_tree_cv,
    bagging_default_cv,
    bagging_subspace_cv,
    bagging_50features_cv,
    bagging_30features_cv,
    rf_cv,
    xgb_cv,
    lgb_cv,
]

results_df = (
    pd.DataFrame([{k: v for k, v in r.items() if k != '_f1_scores'}
                  for r in all_results])
    .sort_values('F1_mean', ascending=False)
    .reset_index(drop=True)
)

print('\n5-FOLD CROSS-VALIDATION SUMMARY')
print(results_df[['Model', 'F1_mean', 'F1_std', 'Accuracy_mean',
                   'Precision_mean', 'Recall_mean']].to_string(index=False))

# Statistical Significance Test (t-test)
def corrected_resampled_ttest(scores_a, scores_b, n_train, n_test):
    k        = len(scores_a)
    diff     = scores_a - scores_b
    d_bar    = diff.mean()
    s_sq     = diff.var(ddof=1)
    var_corr = (1/k + n_test/n_train) * s_sq
    t_stat   = d_bar / np.sqrt(var_corr)
    p_value  = 2 * (1 - stats.t.cdf(abs(t_stat), df=k-1))
    return float(t_stat), float(p_value)

n_total      = len(X_scaled)
n_test_fold  = n_total // N_SPLITS
n_train_fold = n_total - n_test_fold

result_map   = {r['Model']: r['_f1_scores'] for r in all_results}
best_model_name = results_df.iloc[0]['Model']
best_scores = result_map[best_model_name]

print('\nSTATISTICAL SIGNIFICANCE TESTS vs Best Model')
for r in all_results:
    if r['Model'] == best_model_name:
        continue
    t, p = corrected_resampled_ttest(
        r['_f1_scores'], best_scores, n_train_fold, n_test_fold
    )
    print(f'  {r["Model"]:<35}  t={t:+.3f}  p={p:.4f}')

# Save the best model
model_objects = {
    'Single Decision Tree': single_tree,
    'Bagging (All Features)': bagging_default,
    'Bagging (70% Features)': bagging_subspace,
    'Bagging (50% Features)': bagging_50features,
    'Bagging (30% Features)': bagging_30features,
    'Random Forest': rf_model,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model,
}

best_name = results_df.iloc[0]['Model']
best_model = model_objects[best_name]

print(f'\nBEST MODEL: {best_name}')
print(f'CV F1 : {results_df.iloc[0]["F1_mean"]:.4f} +/- {results_df.iloc[0]["F1_std"]:.4f}')

# Train final model on all data
best_model.fit(X_scaled, y_encoded)

# Save model artifact
artifact = {
    'model'                  : best_model,
    'model_name'             : best_name,
    'scaler'                 : scaler,
    'label_encoder'          : label_encoder,
    'feature_columns'        : feature_columns,
    'cv_metrics': {
        'f1_mean'      : float(results_df.iloc[0]['F1_mean']),
        'f1_std'       : float(results_df.iloc[0]['F1_std']),
        'accuracy_mean': float(results_df.iloc[0]['Accuracy_mean']),
    },
    'a4_champion_f1' : CHAMPION_F1,
}

out_path = OUT_DIR / 'bagging_trees_champion.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(artifact, f)

print(f'\nSaved: {out_path}')

# Print feature importances for the best ensemble model
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f'\nTop 10 Most Important Features ({best_name}):')
    for i in range(min(10, len(feature_columns))):
        print(f'  {i+1}. {feature_columns[indices[i]]}: {importances[indices[i]]:.4f}')


"""
T-TEST: LightGBM (A5b classification_bagging_trees.py) vs Baseline Models
======================================================================

LightGBM F1 per fold:     [0.60119669 0.62564327 0.60350582 0.68353009 0.65164652]
LightGBM mean:            0.633104478

A4 Champion RF F1 per fold: [0.59125024 0.62187    0.56044242 0.65402408 0.60242416]
A4 Champion RF mean:      0.6060021800000001

Soft Voting F1 per fold:  [0.61809316 0.63567163 0.61791823 0.69205568 0.65414995]
Soft Voting mean:         0.64357773

Test 1: LightGBM vs A4 Champion Random Forest
  t-statistic: +2.0288
  p-value:     0.1124
  Significant at α=0.05: No

Test 2: LightGBM vs Soft Voting Ensemble
  t-statistic: -2.8028
  p-value:     0.0487
  Significant at α=0.05: Yes

Test 3: Soft Voting vs A4 Champion Random Forest (for reference)
  t-statistic: +3.1372
  p-value:     0.0349
  Significant at α=0.05: Yes

======================================================================
SUMMARY
======================================================================
The LightGBM model from classification_bagging_trees.py shows:
  - Mean F1: 0.6331 +/- 0.0311
  - Compared to Soft Voting (best baseline): t=+3.137, p=0.0349
  - No statistically significant difference (p > 0.05)
"""
