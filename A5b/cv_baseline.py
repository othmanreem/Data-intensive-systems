import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

from sklearn.model_selection import (
    StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f'Total samples      : {X.shape[0]}')

cv_strategy = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)

def evaluate_cv(model, X, y, cv, name='Model'):
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

rf_champion = RandomForestClassifier(
    n_estimators=200, max_depth=15,
    min_samples_split=5, min_samples_leaf=2,
    class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)
champ_cv = evaluate_cv(
    rf_champion, X_scaled, y, cv_strategy,
    name='A4 Champion – Random Forest'
)

soft_voting = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, class_weight='balanced_subsample',
                                       random_state=RANDOM_STATE, n_jobs=-1)),
        ('lr',  LogisticRegression( max_iter=1000, class_weight='balanced',random_state=RANDOM_STATE)),
        ('xgb', xgb.XGBClassifier(  n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, random_state=RANDOM_STATE,class_weight='balanced', n_jobs=-1 )),
        ('lgb', lgb.LGBMClassifier( n_estimators=200, learning_rate=0.1, class_weight='balanced',subsample=0.8, colsample_bytree=0.8,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1 )),
        ('knn', KNeighborsClassifier(n_neighbors=7)),
        ('lda', LinearDiscriminantAnalysis()),
    ],
    voting='soft',
    n_jobs=-1,
)

sv_cv = evaluate_cv(soft_voting, X_scaled, y, cv_strategy, name='Soft Voting')

all_results = [champ_cv, sv_cv]
results_df  = (
    pd.DataFrame([{k: v for k, v in r.items() if k != '_f1_scores'}
                  for r in all_results])
    .sort_values('F1_mean', ascending=False)
    .reset_index(drop=True)
)

print('5-FOLD CROSS-VALIDATION SUMMARY')
print(results_df[['Model','F1_mean','F1_std','Accuracy_mean',
                   'Precision_mean','Recall_mean']].to_string(index=False))

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
champ_scores = result_map['A4 Champion – Random Forest']

print('STATISTICAL SIGNIFICANCE TESTS vs A4 Champion')
for r in all_results:
    if 'Champion' in r['Model']:
        continue
    t, p = corrected_resampled_ttest(
        r['_f1_scores'], champ_scores, n_train_fold, n_test_fold
    )
    print(f'  {r["Model"]:<35}  t={t:+.3f}  p={p:.4f}')

# Save model using cross-validation (fit on all data)
model_objects = {
    'Soft Voting'                : soft_voting,
    'A4 Champion – Random Forest': rf_champion,
}

best_name  = results_df.iloc[0]['Model']
best_model = model_objects[best_name]

print(f'CHAMPION ENSEMBLE: {best_name}')
print(f'CV F1 : {results_df.iloc[0]["F1_mean"]:.4f} +/- {results_df.iloc[0]["F1_std"]:.4f}')

# Fit best model on all data for final deployment
best_model.fit(X_scaled, y)

# Save model
artifact = {
    'model'                  : best_model,
    'model_name'             : best_name,
    'scaler'                 : scaler,
    'feature_columns'        : feature_columns,
    'cv_metrics': {
        'f1_mean'      : float(results_df.iloc[0]['F1_mean']),
        'f1_std'       : float(results_df.iloc[0]['F1_std']),
        'accuracy_mean': float(results_df.iloc[0]['Accuracy_mean']),
        'precision_mean': float(results_df.iloc[0]['Precision_mean']),
        'recall_mean'  : float(results_df.iloc[0]['Recall_mean']),
    },
    'a4_champion_f1' : CHAMPION_F1,
}

out_path = OUT_DIR / 'ensemble_classification_champion.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(artifact, f)

print(f'Saved: {out_path}')
