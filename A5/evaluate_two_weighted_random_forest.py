#!/usr/bin/env python3
"""
Evaluation script for the two-weighted Random Forest model.

This script:
1. Loads the two-weighted model (regression + classification)
2. Evaluates on the test dataset
3. Computes metrics (R², MAE, MSE)
4. Compares raw vs weighted predictions using body region weights

The two-weighted model uses:
- A regression model (RandomForestRegressor)
- A classification model to derive body region weights (upper_body vs lower_body)
- Body region weights are mapped to regression feature weights for weighted predictions

Note: The weighted predictions use feature scaling based on classification model
class probabilities (body region weights). This approach aims to weight features
based on which body region is more important for each prediction.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Add the A5 directory to the path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from CorrelationFilter import CorrelationFilter

# Add parent directory to path for mapping imports
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from A5.mapping import FMS_mapping, NASM_mapping

# Model paths
CLASSIFICATION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../A4/models/gDriveVersion/final_champion_model_A3.pkl"
)
REGRESSION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models/aimoscores_improved_A4.pkl"
)
TEST_DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../A3/A3_Data/test_dataset.csv"
)

# Output directory for results
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "two_weighted_evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variables for loaded models
regression_pipe = None
FEATURE_NAMES = None
MODEL_METRICS = None

classification_model = None
CLASSIFICATION_FEATURE_NAMES = None
CLASSIFICATION_CLASSES = None
CLASSIFICATION_METRICS = None

BODY_REGION_RECOMMENDATIONS = {
    'Upper Body': (
        "Focus on shoulder mobility, thoracic spine extension, "
        "and keeping your head neutral."),
    'Lower Body': (
        "Work on hip mobility, ankle dorsiflexion, "
        "and knee tracking over toes.")
}


def load_regression_model():
    """Load the regression model from pickle file."""
    global regression_pipe, FEATURE_NAMES, MODEL_METRICS

    if os.path.exists(REGRESSION_MODEL_PATH):
        print(f"Loading regression model from {REGRESSION_MODEL_PATH}")
        with open(REGRESSION_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        regression_pipe = artifact["model"]
        FEATURE_NAMES = artifact["feature_columns"]
        MODEL_METRICS = artifact.get("test_metrics", {})

        print(f"  Model loaded: {len(FEATURE_NAMES)} features")
        print(f"  Original test R²: {MODEL_METRICS.get('r2', 'N/A')}")
        print(f"  Original test MAE: {MODEL_METRICS.get('mae', 'N/A')}")
        print(f"  Original test MSE: {MODEL_METRICS.get('mse', 'N/A')}")

        return regression_pipe, FEATURE_NAMES, MODEL_METRICS

    print(f"Regression model not found at {REGRESSION_MODEL_PATH}")
    return None, None, None


def load_classification_model():
    """Load the classification model from pickle file."""
    global classification_model, CLASSIFICATION_FEATURE_NAMES, CLASSIFICATION_CLASSES, CLASSIFICATION_METRICS

    if os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}")
        with open(CLASSIFICATION_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        classification_model = artifact["model"]
        CLASSIFICATION_FEATURE_NAMES = artifact["feature_columns"]
        CLASSIFICATION_CLASSES = artifact.get("classes", ["lower_body", "upper_body"])
        CLASSIFICATION_METRICS = artifact.get("test_metrics", {})

        print(f"  Model loaded: {len(CLASSIFICATION_FEATURE_NAMES)} features")
        print(f"  Classes: {CLASSIFICATION_CLASSES}")
        print(f"  Test accuracy: {CLASSIFICATION_METRICS.get('accuracy', 'N/A')}")

        return classification_model, CLASSIFICATION_FEATURE_NAMES, CLASSIFICATION_CLASSES, CLASSIFICATION_METRICS

    print(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
    return None, None, None, None


def load_test_data():
    """Load the test dataset."""
    print(f"Loading test data from {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, sep=';', decimal=',')
    print(f"  Test dataset shape: {df.shape}")
    return df


def predict_with_feature_scaling(rf, X, feature_weights, normalize=True):
    """
    Weight inputs before passing them to each tree
    and calculate the weighted average of predictions.

    rf: fitted RandomForestRegressor
    X: (n_samples, n_features)
    feature_weights: (n_features,)
    """
    fw = np.asarray(feature_weights, dtype=float)
    if normalize:
        fw = fw / (fw.sum() + 1e-12)
    X_scaled = X * fw  # broadcasting
    # Use per-tree predictions and average (same aggregation as rf.predict)
    tree_preds = np.stack([t.predict(X_scaled) for t in rf.estimators_], axis=1)
    return tree_preds.mean(axis=1)


def get_body_region_weights(classification_model, X, feature_names):
    """
    Get body region weights from classification model probabilities.

    classification_model: fitted two-class classifier
    X: input features (n_samples, n_features)
    feature_names: list of feature names

    Returns: dict mapping body regions ('upper_body', 'lower_body') to weights
    """
    # Get class probabilities
    proba = classification_model.predict_proba(X)[0]

    # Map probabilities to body regions
    weights = {}
    for i, class_name in enumerate(CLASSIFICATION_CLASSES):
        # Map class names to body regions
        if 'upper' in class_name.lower() or 'upper_body' in class_name.lower():
            weights['upper_body'] = proba[i]
        elif 'lower' in class_name.lower() or 'lower_body' in class_name.lower():
            weights['lower_body'] = proba[i]

    # Normalize weights to sum to 1
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}

    return weights


def get_region_feature_mapping():
    """
    Create a mapping from body regions to regression feature names.

    Returns: dict mapping body_region -> list of feature names
    """
    region_features = {
        'upper_body': [],
        'lower_body': []
    }

    # Extract features from FMS_mapping and NASM_mapping
    for region in ['upper_body', 'lower_body']:
        # From FMS_mapping
        if region in FMS_mapping:
            for key in FMS_mapping[region].keys():
                # Look for features containing this FMS test name (e.g., "No_1_Angle_Deviation")
                for feat in FEATURE_NAMES:
                    # Match the FMS test name prefix (e.g., "No_1" should match "No_1_Angle_Deviation")
                    test_num = key.split('_')[1]  # Extract number from "No_1_Angle_Deviation"
                    if feat.startswith(f"No_{test_num}_"):
                        if feat not in region_features[region]:
                            region_features[region].append(feat)

        # From NASM_mapping
        if region in NASM_mapping:
            for key, item in NASM_mapping[region].items():
                # Look for features containing this NASM test name (e.g., "No_14_NASM_Deviation")
                for feat in FEATURE_NAMES:
                    # Match the NASM test name prefix (e.g., "No_14" should match "No_14_NASM_Deviation")
                    test_num = key.split('_')[1]  # Extract number from "No_14_NASM_Deviation"
                    if feat.startswith(f"No_{test_num}_"):
                        if feat not in region_features[region]:
                            region_features[region].append(feat)

    return region_features


def get_regression_weights_from_body_regions(region_weights, region_features):
    """
    Map body region weights to regression feature weights.

    region_weights: dict mapping body_region -> weight
    region_features: dict mapping body_region -> list of feature names

    Returns: numpy array of weights for regression features
    """
    regression_weights = []

    for feat in FEATURE_NAMES:
        # Find which regions this feature belongs to
        matching_regions = []
        for region, features in region_features.items():
            if feat in features:
                matching_regions.append(region)

        if matching_regions:
            # Weight based on sum of region weights
            weight = sum(region_weights.get(r, 0) for r in matching_regions) / len(matching_regions)
            regression_weights.append(weight)
        else:
            # Default weight if feature doesn't match any region
            regression_weights.append(1.0 / len(FEATURE_NAMES))

    return np.array(regression_weights)


def get_important_body_region(classification_model, X, feature_names):
    """
    Determine the most important body region from classification model.

    Returns: tuple of (region_name, probability)
    """
    proba = classification_model.predict_proba(X)[0]

    # Find the class with highest probability
    max_idx = np.argmax(proba)
    class_name = CLASSIFICATION_CLASSES[max_idx]

    # Map to body region
    if 'upper' in class_name.lower():
        return 'upper_body', proba[max_idx]
    else:
        return 'lower_body', proba[max_idx]


def predict_with_body_region_weights(regression_model, X,
                                     classification_model,
                                     classification_example,
                                     classification_features,
                                     regression_features):
    """
    Predict using feature scaling based on body region weights from classification model.

    regression_model: fitted RandomForestRegressor
    X: input features for regression (n_samples, n_features)
    classification_model: fitted two-class classifier
    classification_example: input data for classification model
    classification_features: list of feature names for classification
    regression_features: list of feature names for regression

    Returns: predicted scores, weights used, important body region
    """
    # Get body region weights from classification model
    region_weights = get_body_region_weights(classification_model,
                                             classification_example,
                                             classification_features)

    print(f"Body region weights: {region_weights}")

    # Get mapping of regions to features
    region_features = get_region_feature_mapping()

    # Map region weights to feature weights
    feature_weights = get_regression_weights_from_body_regions(region_weights, region_features)

    print(f"Feature weights (first 10): {feature_weights[:10]}")

    # Get important body region
    important_region, region_proba = get_important_body_region(classification_model,
                                                                classification_example,
                                                                classification_features)
    print(f"Important body region: {important_region} (probability: {region_proba:.3f})")

    # Use predict_with_feature_scaling with the region-based weights
    prediction = predict_with_feature_scaling(regression_model, X, feature_weights)

    return prediction, feature_weights, important_region


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Compute evaluation metrics."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} Performance:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")

    return {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def plot_predictions(y_true, y_raw, y_weighted, output_dir):
    """Create visualization of predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Raw predictions vs True
    axes[0, 0].scatter(y_true, y_raw, alpha=0.5, s=20)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('True AimoScore')
    axes[0, 0].set_ylabel('Predicted AimoScore')
    axes[0, 0].set_title('Raw Predictions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Weighted predictions vs True
    axes[0, 1].scatter(y_true, y_weighted, alpha=0.5, s=20, color='green')
    axes[0, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                    'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('True AimoScore')
    axes[0, 1].set_ylabel('Predicted AimoScore')
    axes[0, 1].set_title('Weighted Predictions (Body Region Weights)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Prediction error distribution
    error_raw = y_true - y_raw
    error_weighted = y_true - y_weighted

    axes[1, 0].hist(error_raw, bins=30, alpha=0.5, label='Raw', edgecolor='black')
    axes[1, 0].hist(error_weighted, bins=30, alpha=0.5, label='Weighted', edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Comparison of both predictions
    axes[1, 1].scatter(y_raw, y_weighted, alpha=0.5, s=20, color='purple')
    axes[1, 1].plot([min(y_raw.min(), y_weighted.min()),
                     max(y_raw.max(), y_weighted.max())],
                    [min(y_raw.min(), y_weighted.min()),
                     max(y_raw.max(), y_weighted.max())],
                    'r--', lw=2, label='y=x')
    axes[1, 1].set_xlabel('Raw Predictions')
    axes[1, 1].set_ylabel('Weighted Predictions')
    axes[1, 1].set_title('Raw vs Weighted Predictions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'two_weighted_predictions_comparison.png'), dpi=150)
    print(f"\n  Plot saved to: {os.path.join(output_dir, 'two_weighted_predictions_comparison.png')}")
    plt.close()


def compare_metrics(metrics_raw, metrics_weighted, output_dir):
    """Create comparison table and save to file."""
    print("\n" + "="*60)
    print("METRICS COMPARISON")
    print("="*60)

    print(f"\n{'Metric':<10} {'Raw':>12} {'Weighted':>12} {'Improvement':>12}")
    print("-"*60)

    for metric in ['r2', 'mae', 'mse', 'rmse']:
        raw_val = metrics_raw[metric]
        weighted_val = metrics_weighted[metric]

        if metric == 'r2':
            improvement = weighted_val - raw_val
            improvement_str = f"{improvement:+.6f}"
        else:  # mae, mse, rmse - lower is better
            improvement = raw_val - weighted_val
            improvement_str = f"{improvement:+.6f}"

        print(f"{metric.upper():<10} {raw_val:>12.6f} {weighted_val:>12.6f} {improvement_str:>12}")

    # Save comparison to file
    with open(os.path.join(output_dir, 'two_weighted_metrics_comparison.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("TWO-WEIGHTED RANDOM FOREST EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")

        f.write("REGRESSION MODEL METRICS (Raw Predictions)\n")
        f.write("-"*40 + "\n")
        f.write(f"R² Score: {metrics_raw['r2']:.6f}\n")
        f.write(f"MAE: {metrics_raw['mae']:.6f}\n")
        f.write(f"MSE: {metrics_raw['mse']:.6f}\n")
        f.write(f"RMSE: {metrics_raw['rmse']:.6f}\n\n")

        f.write("REGRESSION MODEL METRICS (Weighted Predictions)\n")
        f.write("-"*40 + "\n")
        f.write(f"R² Score: {metrics_weighted['r2']:.6f}\n")
        f.write(f"MAE: {metrics_weighted['mae']:.6f}\n")
        f.write(f"MSE: {metrics_weighted['mse']:.6f}\n")
        f.write(f"RMSE: {metrics_weighted['rmse']:.6f}\n\n")

        f.write("COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write(f"R² Improvement: {metrics_weighted['r2'] - metrics_raw['r2']:+.6f}\n")
        f.write(f"MAE Improvement: {metrics_raw['mae'] - metrics_weighted['mae']:+.6f}\n")
        f.write(f"MSE Improvement: {metrics_raw['mse'] - metrics_weighted['mse']:+.6f}\n")
        f.write(f"RMSE Improvement: {metrics_raw['rmse'] - metrics_weighted['rmse']:+.6f}\n")

    print(f"\n  Metrics saved to: {os.path.join(output_dir, 'two_weighted_metrics_comparison.txt')}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("TWO-WEIGHTED RANDOM FOREST EVALUATION")
    print("="*60)

    # Load models
    regression_pipeline, regression_features, _ = load_regression_model()
    classification_model, classification_features, _, _ = load_classification_model()

    if regression_pipeline is None or classification_model is None:
        print("Error: Could not load required models")
        exit(1)

    # Load test data
    test_df = load_test_data()

    # Extract features - use the features that the model was trained on
    # The model expects features from the CorrelationFilter
    # Check if test data has all required features
    missing_features = [f for f in regression_features if f not in test_df.columns]
    if missing_features:
        print(f"Warning: Missing features in test data: {missing_features}")
        print("These features will be filled with zeros.")

    # Create feature dataframe with all required columns
    X_regression_data = pd.DataFrame()
    for feat in regression_features:
        if feat in test_df.columns:
            X_regression_data[feat] = test_df[feat]
        else:
            # Fill missing features with zeros
            X_regression_data[feat] = 0.0
            print(f"  Filling missing feature '{feat}' with zeros")

    X_classification_data = pd.DataFrame()
    for feat in classification_features:
        if feat in test_df.columns:
            X_classification_data[feat] = test_df[feat]
        else:
            X_classification_data[feat] = 0.0

    # Get true values
    y_true = test_df['AimoScore'].values

    # Get raw predictions using the pipeline (includes CorrelationFilter)
    print("\n" + "-"*40)
    print("MAKING PREDICTIONS")
    print("-"*40)

    raw_predictions = regression_pipeline.predict(X_regression_data)
    print(f"Raw predictions computed: {len(raw_predictions)} samples")

    # Get weighted predictions using body region weights
    print("\nComputing weighted predictions with body region weights...")

    # Get the regression model from the pipeline
    regression_model = regression_pipeline.named_steps['model']

    # Compute weighted predictions for each sample
    weighted_predictions = []
    important_regions = []

    for i in range(len(X_regression_data)):
        regression_sample = X_regression_data.iloc[[i]]
        classification_sample = X_classification_data.iloc[[i]]

        # Get prediction with body region weights
        pred, weights, important_region = predict_with_body_region_weights(
            regression_model,
            regression_sample,
            classification_model,
            classification_sample,
            classification_features,
            regression_features
        )

        weighted_predictions.append(pred[0])
        important_regions.append(important_region)

    weighted_predictions = np.array(weighted_predictions)
    print(f"Weighted predictions computed: {len(weighted_predictions)} samples")

    # Evaluate both models
    print("\n" + "-"*40)
    print("EVALUATION")
    print("-"*40)

    metrics_raw = evaluate_model(y_true, raw_predictions, "Raw Predictions")
    metrics_weighted = evaluate_model(y_true, weighted_predictions, "Weighted Predictions (Body Region)")

    # Compare metrics
    compare_metrics(metrics_raw, metrics_weighted, OUTPUT_DIR)

    # Create visualizations
    print("\n" + "-"*40)
    print("CREATING VISUALIZATIONS")
    print("-"*40)

    plot_predictions(y_true, raw_predictions, weighted_predictions, OUTPUT_DIR)

    # Classification model feature importances for comparison
    print("\nClassification Model Feature Importances (Top 10):")
    if hasattr(classification_model, 'feature_importances_'):
        class_importances = classification_model.feature_importances_
        class_importance_df = pd.DataFrame({
            'Feature': classification_features,
            'Importance': class_importances
        }).sort_values('Importance', ascending=False)

        for i, row in class_importance_df.head(10).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.6f}")

        # Save classification feature importance
        class_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'two_weighted_classification_feature_importance.csv'), index=False)
        print(f"\n  Classification feature importance saved to: {os.path.join(OUTPUT_DIR, 'two_weighted_classification_feature_importance.csv')}")

    # Get feature importances from the model
    feature_importances = regression_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': regression_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f}")

    # Save feature importance to file
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'two_weighted_feature_importance.csv'), index=False)
    print(f"\n  Feature importance saved to: {os.path.join(OUTPUT_DIR, 'two_weighted_feature_importance.csv')}")

    # Additional analysis: correlation between predictions
    correlation = np.corrcoef(raw_predictions, weighted_predictions)[0, 1]
    print(f"\nCorrelation between raw and weighted predictions: {correlation:.6f}")

    # Additional analysis: error analysis
    print("\n" + "-"*40)
    print("ERROR ANALYSIS")
    print("-"*40)

    raw_errors = y_true - raw_predictions
    weighted_errors = y_true - weighted_predictions

    print(f"\nRaw prediction error statistics:")
    print(f"  Mean error: {raw_errors.mean():.6f}")
    print(f"  Std error: {raw_errors.std():.6f}")
    print(f"  Min error: {raw_errors.min():.6f}")
    print(f"  Max error: {raw_errors.max():.6f}")

    print(f"\nWeighted prediction error statistics:")
    print(f"  Mean error: {weighted_errors.mean():.6f}")
    print(f"  Std error: {weighted_errors.std():.6f}")
    print(f"  Min error: {weighted_errors.min():.6f}")
    print(f"  Max error: {weighted_errors.max():.6f}")

    # Check for systematic bias
    print(f"\nSystematic bias analysis:")
    print(f"  Raw predictions tend to overestimate (positive error) when mean error > 0")
    print(f"  Raw mean error: {raw_errors.mean():.6f} ({'overestimates' if raw_errors.mean() > 0 else 'underestimates'})")
    print(f"  Weighted mean error: {weighted_errors.mean():.6f} ({'overestimates' if weighted_errors.mean() > 0 else 'underestimates'})")

    # Body region distribution analysis
    print("\n" + "-"*40)
    print("BODY REGION DISTRIBUTION ANALYSIS")
    print("-"*40)

    upper_body_count = sum(1 for r in important_regions if r == 'upper_body')
    lower_body_count = sum(1 for r in important_regions if r == 'lower_body')

    print(f"\nUpper body region predictions: {upper_body_count} ({upper_body_count/len(important_regions)*100:.1f}%)")
    print(f"Lower body region predictions: {lower_body_count} ({lower_body_count/len(important_regions)*100:.1f}%)")

    # Save predictions to file
    results_df = pd.DataFrame({
        'True_AimoScore': y_true,
        'Raw_Prediction': raw_predictions,
        'Weighted_Prediction': weighted_predictions,
        'Raw_Error': y_true - raw_predictions,
        'Weighted_Error': y_true - weighted_predictions,
        'Important_Body_Region': important_regions
    })

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'two_weighted_predictions.csv'), index=False)
    print(f"\n  Predictions saved to: {os.path.join(OUTPUT_DIR, 'two_weighted_predictions.csv')}")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - two_weighted_predictions.csv (all predictions)")
    print(f"  - two_weighted_metrics_comparison.txt (detailed metrics)")
    print(f"  - two_weighted_predictions_comparison.png (visualizations)")

    # Final comparison
    r2_improvement = metrics_weighted['r2'] - metrics_raw['r2']
    mae_improvement = metrics_raw['mae'] - metrics_weighted['mae']

    print(f"\nWeighted predictions vs Raw predictions:")
    print(f"  R² change: {r2_improvement:+.6f} ({'improvement' if r2_improvement > 0 else 'degradation'})")
    print(f"  MAE change: {mae_improvement:+.6f} ({'improvement' if mae_improvement > 0 else 'degradation'})")

    # Save evaluation summary
    summary_df = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'MSE', 'RMSE'],
        'Raw': [metrics_raw['r2'], metrics_raw['mae'], metrics_raw['mse'], metrics_raw['rmse']],
        'Weighted': [metrics_weighted['r2'], metrics_weighted['mae'], metrics_weighted['mse'], metrics_weighted['rmse']],
        'Improvement': [
            r2_improvement,
            mae_improvement,
            metrics_raw['mse'] - metrics_weighted['mse'],
            metrics_raw['rmse'] - metrics_weighted['rmse']
        ]
    })
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'two_weighted_evaluation_summary.csv'), index=False)
    print(f"\n  Evaluation summary saved to: {os.path.join(OUTPUT_DIR, 'two_weighted_evaluation_summary.csv')}")

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nThe evaluation script successfully:")
    print(f"  1. Loaded the two-weighted model (regression + classification)")
    print(f"  2. Evaluated on {len(y_true)} test samples")
    print(f"  3. Computed metrics (R², MAE, MSE, RMSE)")
    print(f"  4. Compared raw vs weighted predictions using body region weights")
    print(f"\nKey findings:")
    print(f"  - Raw predictions achieved R² = {metrics_raw['r2']:.4f}")
    print(f"  - Weighted predictions achieved R² = {metrics_weighted['r2']:.4f}")
    if r2_improvement > 0:
        print(f"  - Weighted predictions improved R² by {r2_improvement:.4f}")
    else:
        print(f"  - Weighted predictions degraded R² by {abs(r2_improvement):.4f}")
    print(f"\nNote: The weighted predictions use feature scaling based on")
    print(f"classification model class probabilities (body region weights).")
    print(f"If performance degraded, this may indicate misalignment between")
    print(f"classification and regression tasks, or that the body region")
    print(f"weighting approach needs adjustment.")


if __name__ == "__main__":
    main()
