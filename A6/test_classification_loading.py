#!/usr/bin/env python3
"""
Script to load and execute all classification models with one sample.
Tests models from A4, A5, A5b, and A6.

Data loading adapted from classification_baseline.py to use the same
data processing pipeline for consistent feature extraction.

NOTE: A4 Random Forest model was trained WITH the 5 duplicate NASM columns
(No_1_NASM_Deviation through No_5_NASM_Deviation), while other models (A5, A5b, A6)
were trained WITHOUT them. This script loads data WITH the duplicate columns
to support the A4 model, and filters them out for other models as needed.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import model paths from all_classification.py
sys.path.insert(0, project_root)
from all_classification import (
    a4_rf,
    a5_ensemnble,
    a5b_adaboost,
    a5b_bagging_tree,
    a6_svm
)

# Import custom classes from A5b classification_adaboost.py
# These are needed for unpickling the AdaBoost model
#sys.path.insert(0, os.path.join(project_root, '..', 'A5b'))
from adaboost_classes import (
    AdaBoostEnsemble,
    WeightedDecisionTree
)

# Data paths
REPO_ROOT    = os.path.abspath(os.path.join(project_root, '..'))
DATA_DIR     = os.path.join(REPO_ROOT, 'Datasets_all')

# Weaklink categories (14 classes)
WEAKLINK_CATEGORIES = [
    'ExcessiveForwardLean', 'ForwardHead', 'LeftArmFallForward',
    'LeftAsymmetricalWeightShift', 'LeftHeelRises', 'LeftKneeMovesInward',
    'LeftKneeMovesOutward', 'LeftShoulderElevation', 'RightArmFallForward',
    'RightAsymmetricalWeightShift', 'RightHeelRises', 'RightKneeMovesInward',
    'RightKneeMovesOutward', 'RightShoulderElevation'
]

# Duplicate NASM columns to remove (as in classification_baseline.py)
# NOTE: A4 Random Forest model was trained WITH these 5 duplicate columns,
# so they must be kept in the data for A4 to work correctly
DUPLICATE_NASM_COLS = [
    'No_1_NASM_Deviation',
    'No_2_NASM_Deviation',
    'No_3_NASM_Deviation',
    'No_4_NASM_Deviation',
    'No_5_NASM_Deviation',
]

# Columns to exclude when extracting features
EXCLUDE_COLS = ['ID', 'WeakestLink', 'EstimatedScore']

# Expected classification classes (14 weaklink categories)
EXPECTED_CLASSES = [
    'ExcessiveForwardLean', 'ForwardHead', 'LeftArmFallForward',
    'LeftAsymmetricalWeightShift', 'LeftHeelRises', 'LeftKneeMovesInward',
    'LeftKneeMovesOutward', 'LeftShoulderElevation', 'RightArmFallForward',
    'RightAsymmetricalWeightShift', 'RightHeelRises', 'RightKneeMovesInward',
    'RightKneeMovesOutward', 'RightShoulderElevation'
]


def load_and_prepare_data():
    """Load and prepare data following the same pipeline as classification_baseline.py.

    NOTE: This function loads data WITH the 5 duplicate NASM columns because
    the A4 Random Forest model was trained with those columns included.
    Other models (A5, A5b, A6) will filter out these columns based on their feature_columns.
    """
    # Load datasets
    movement_features_df = pd.read_csv(os.path.join(DATA_DIR, 'aimoscores.csv'))
    weaklink_scores_df = pd.read_csv(os.path.join(DATA_DIR, 'scores_and_weaklink.csv'))

    print('Movement features shape:', movement_features_df.shape)
    print('Weak link scores shape:', weaklink_scores_df.shape)

    # NOTE: We do NOT remove duplicate NASM columns here because
    # the A4 Random Forest model was trained WITH these columns
    # The other models (A5, A5b, A6) will filter them out based on their saved feature_columns
    print('NOTE: Keeping duplicate NASM columns for A4 Random Forest model compatibility')

    # Create WeakestLink target column
    weaklink_scores_df['WeakestLink'] = (
        weaklink_scores_df[WEAKLINK_CATEGORIES].idxmax(axis=1)
    )
    print('Weakest Link class distribution:')
    print(weaklink_scores_df['WeakestLink'].value_counts())

    # Merge datasets
    target_df = weaklink_scores_df[['ID', 'WeakestLink']].copy()
    merged_df = movement_features_df.merge(target_df, on='ID', how='inner')
    print('Merged dataset shape:', merged_df.shape)

    # Extract feature columns - include ALL columns except EXCLUDE_COLS
    # This ensures the 5 duplicate NASM columns are included for A4
    feature_columns = [c for c in merged_df.columns if c not in EXCLUDE_COLS]

    X = merged_df[feature_columns].values
    y = merged_df['WeakestLink'].values

    print(f'Feature matrix shape : {X.shape}')
    print(f'Number of features   : {len(feature_columns)}')
    print(f'Number of classes    : {len(np.unique(y))}')

    # Create train/test split (same as baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'feature_columns': feature_columns,
        'scaler': scaler,
        'X_train': X_train,
        'X_train_scaled': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'merged_df': merged_df,
    }


def load_model(model_path, model_name):
    """Load a model from a pickle file."""
    full_path = os.path.join(project_root, model_path)

    if not os.path.exists(full_path):
        print(f"  ⚠️  Model file not found: {full_path}")
        return None, None, None, None

    try:
        with open(full_path, 'rb') as f:
            artifact = pickle.load(f)

        # Extract model and scaler based on artifact structure
        if isinstance(artifact, dict):
            model = artifact.get('model')
            scaler = artifact.get('scaler')
            feature_columns = artifact.get('feature_columns')
        else:
            # A6 SVM is a Pipeline object
            model = artifact
            # Extract scaler from pipeline if it exists
            if hasattr(model, 'steps') and len(model.steps) >= 1:
                # Find the scaler in the pipeline
                scaler = None
                for step_name, step_obj in model.steps:
                    if hasattr(step_obj, 'transform'):
                        # Check if this is a scaler (has n_features_in_ attribute)
                        if hasattr(step_obj, 'n_features_in_') and not hasattr(step_obj, 'predict'):
                            scaler = step_obj
                            break
                # If no scaler found, try to get it from the first step
                if scaler is None and len(model.steps) > 0:
                    first_step = model.steps[0][1]
                    if hasattr(first_step, 'transform') and hasattr(first_step, 'n_features_in_'):
                        scaler = first_step
            # For A6 SVM pipeline, extract feature columns from the scaler
            feature_columns = None
            if hasattr(model, 'steps') and len(model.steps) > 0:
                # Get feature names from the first step (should be the scaler)
                first_step = model.steps[0][1]
                if hasattr(first_step, 'get_feature_names_out'):
                    try:
                        names = first_step.get_feature_names_out()
                        # Only use feature names if they are real column names,
                        # not generic placeholder names like x0, x1, ...
                        import re
                        if not all(re.fullmatch(r'x\d+', n) for n in names):
                            feature_columns = names
                        # else: leave feature_columns = None; handled below
                    except:
                        pass

        print(f"  ✓ Loaded {model_name}")
        #print(model, scaler, feature_columns, artifact)
        return model, scaler, feature_columns, artifact
    except Exception as e:
        print(f"  ✗ Error loading {model_name}: {e}")
        return None, None, None, None


def predict_with_model(model, scaler, sample_features, model_name):
    """Make a prediction using the model."""
    try:
        features = sample_features.copy()

        # Apply scaler if available
        if scaler is not None:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)
        prediction_proba = None

        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features)

        return prediction, prediction_proba, None
    except Exception as e:
        return None, None, str(e)


def create_sample_from_training_data(training_data, feature_columns, scaler):
    """Create a sample from the training data for testing."""
    # Get first sample from training data
    sample = training_data['X_train'][0:1].copy()
    sample_df = pd.DataFrame(sample, columns=feature_columns)

    # Scale if scaler is available
    if scaler is not None:
        sample_df_scaled = scaler.transform(sample_df)
        return sample_df, sample_df_scaled
    return sample_df, sample_df


def filter_features_for_model(sample_df, model_feature_columns):
    """Filter sample data to only include features the model expects."""
    available_features = [f for f in model_feature_columns if f in sample_df.columns]

    if len(available_features) == 0:
        print(f"  ⚠️  No matching features found, using all available")
        available_features = sample_df.columns.tolist()

    return sample_df[available_features]


def main():
    """Main function to test all models."""
    print("=" * 60)
    print("Testing All Classification Models with One Sample")
    print("=" * 60)
    print()

    # Load and prepare data using the same pipeline as classification_baseline.py
    # NOTE: Data is loaded WITH the 5 duplicate NASM columns for A4 compatibility
    print("Loading data...")
    data = load_and_prepare_data()
    print()

    # Create sample from training data
    sample_features, sample_features_scaled = create_sample_from_training_data(
        data, data['feature_columns'], data['scaler']
    )
    print(f"Sample data shape: {sample_features.shape}")
    print(f"Number of features (including duplicates): {len(data['feature_columns'])}")
    print()

    # Define models to test
    models_to_test = [
        ('A4 Random Forest', a4_rf),
        ('A5 Ensemble', a5_ensemnble),
        ('A5b Adaboost', a5b_adaboost),
        ('A5b Bagging Trees', a5b_bagging_tree),
        ('A6 SVM', a6_svm),
    ]

    results = []

    for model_name, model_path in models_to_test:
        print(f"Testing {model_name}...")

        # Load model
        model, scaler, model_feature_columns, artifact = load_model(model_path, model_name)

        if model is None:
            print(f"  Skipping {model_name} due to load error")
            results.append((model_name, 'LOAD_ERROR', None, None, None))
            print()
            continue

        # Determine feature columns to use
        if model_feature_columns is not None:
            # Filter sample data to only include features the model expects
            test_features = filter_features_for_model(sample_features, model_feature_columns)
            print(f"  Model expects {len(model_feature_columns)} features, using {len(test_features.columns)} available")
        elif hasattr(model, 'steps'):
            # Pipeline with generic/unknown feature names (e.g. A6 SVM trained without
            # the 5 duplicate NASM columns). Drop those duplicate columns so the number
            # of features matches what the pipeline's scaler expects.
            first_step = model.steps[0][1]
            n_expected = getattr(first_step, 'n_features_in_', None)
            cols_without_dupes = [c for c in sample_features.columns if c not in DUPLICATE_NASM_COLS]
            if n_expected is not None and len(cols_without_dupes) == n_expected:
                test_features = sample_features[cols_without_dupes]
                print(f"  Pipeline expects {n_expected} features — dropped duplicate NASM cols, using {len(test_features.columns)} features")
            else:
                # Fallback: just take the first n_expected columns
                test_features = sample_features.iloc[:, :n_expected] if n_expected else sample_features
                print(f"  Pipeline expects {n_expected} features, sliced sample to {len(test_features.columns)} features")
        else:
            test_features = sample_features
            print(f"  Using all {len(sample_features.columns)} available features")

        # Make prediction
        # For A6 SVM pipeline, don't pass the scaler separately since it's already in the pipeline
        # For other models, pass the scaler if available
        if model_feature_columns is None and hasattr(model, 'steps'):
            # This is likely the A6 SVM pipeline - don't apply scaler separately
            scaler_to_use = None
        else:
            scaler_to_use = scaler

        prediction, prediction_proba, error = predict_with_model(
            model, scaler_to_use, test_features, model_name
        )

        if error:
            print(f"  ✗ Prediction error: {error}")
            results.append((model_name, 'PREDICTION_ERROR', None, None, error))
            print()
            continue

        # Display results
        print(f"  ✓ Prediction: {prediction[0]}")

        if prediction_proba is not None:
            print(f"  ✓ Prediction probabilities shape: {prediction_proba.shape}")
            top_classes_idx = np.argsort(prediction_proba[0])[-3:][::-1]
            top_classes = [EXPECTED_CLASSES[i] for i in top_classes_idx]
            top_probs = [prediction_proba[0][i] for i in top_classes_idx]
            print(f"  ✓ Top 3 classes: {list(zip(top_classes, [f'{p:.3f}' for p in top_probs]))}")

        print(f"  ✓ Model type: {type(model).__name__}")

        # Check if model has classes attribute
        if hasattr(model, 'classes_'):
            print(f"  ✓ Model classes: {list(model.classes_)}")

        results.append((model_name, 'SUCCESS', prediction, prediction_proba, None))
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    for model_name, status, prediction, proba, error in results:
        if status == 'SUCCESS':
            pred_str = prediction[0] if prediction is not None else 'N/A'
            print(f"  {model_name}: ✓ SUCCESS - Prediction: {pred_str}")
        else:
            print(f"  {model_name}: ✗ {status} - {error}")

    print()
    print("All models tested!")


if __name__ == "__main__":
    main()
