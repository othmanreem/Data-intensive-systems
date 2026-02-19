#!/usr/bin/env python3
"""
Evaluation script for the stacked Random Forest model.

This script:
1. Loads the stacked model (regression + classification)
2. Evaluates on the test dataset
3. Computes metrics (R², MAE, MSE)
4. Compares raw vs weighted predictions
+
The stacked model uses:
- A regression pipeline with CorrelationFilter and RandomForestRegressor
- A classification model to derive feature weights for weighted predictions
+
Note: The weighted predictions use feature scaling based on classification model
feature importances. If this approach doesn't improve performance, it may indicate
that the classification model's feature priorities don't align well with the
regression task, or that the feature scaling approach needs adjustment.
 """

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Add the A5 directory to the path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from CorrelationFilter import CorrelationFilter

# Model paths
REGRESSION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models/aimoscores_improved_A4.pkl"
)
CLASSIFICATION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models/weaklink_classifier_rfc_A4.pkl"
)
TEST_DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../A3/A3_Data/test_dataset.csv"
)

# Output directory for results
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_regression_model():
    """Load the regression model from pickle file."""
    print(f"Loading regression model from {REGRESSION_MODEL_PATH}")
    with open(REGRESSION_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_names = artifact["feature_columns"]
    metrics = artifact.get("test_metrics", {})

    print(f"  Model loaded: {len(feature_names)} features")
    print(f"  Original test R²: {metrics.get('r2', 'N/A')}")
    print(f"  Original test MAE: {metrics.get('mae', 'N/A')}")
    print(f"  Original test MSE: {metrics.get('mse', 'N/A')}")

    return model, feature_names, metrics


def load_classification_model():
    """Load the classification model from pickle file."""
    print(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}")
    with open(CLASSIFICATION_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    feature_names = artifact["feature_columns"]
    classes = artifact.get("classes", [])
    metrics = artifact.get("test_metrics", {})

    print(f"  Model loaded: {len(feature_names)} features")
    print(f"  Classes: {classes}")
    print(f"  Test accuracy: {metrics.get('accuracy', 'N/A')}")

    return model, feature_names, classes, metrics


def load_test_data():
    """Load the test dataset."""
    print(f"Loading test data from {TEST_DATA_PATH}")
    df = pd.read_csv(TEST_DATA_PATH, sep=';', decimal=',')
    print(f"  Test dataset shape: {df.shape}")
    return df


def get_regression_features(df, regression_features):
    """Extract regression features from dataframe."""
    X = df[regression_features].values
    return X


def get_classification_features(df, classification_features):
    """Extract classification features from dataframe."""
    X = df[classification_features].values
    return X


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


def get_classification_weights(classification_model, feature_names):
    """
    Get feature weights from classification model.

    Uses the classification model's feature importances.

    classification_model: fitted RandomForestClassifier
    feature_names: list of feature names

    Returns: dict mapping feature names to weights
    """
    if hasattr(classification_model, 'feature_importances_'):
        importances = classification_model.feature_importances_
        weights = dict(zip(feature_names, importances))

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights

    # Fallback: equal weights
    return {f: 1.0/len(feature_names) for f in feature_names}


def get_regression_weights_from_classification(classification_model,
                                               classification_features,
                                               regression_features):
    """
    Map classification model weights to regression feature space.

    classification_model: fitted RandomForestClassifier
    classification_features: list of feature names for classification
    regression_features: list of feature names for regression

    Returns: numpy array of weights for regression features
    """
    # Get weights from classification model
    class_weights = get_classification_weights(classification_model,
                                               classification_features)

    # Map to regression features
    regression_weights = []
    for feat in regression_features:
        # Try to find matching feature in classification features
        if feat in class_weights:
            regression_weights.append(class_weights[feat])
        else:
            # Check for partial matches (e.g., "No_1_Angle_Deviation" matches "No_1_NASM_Deviation")
            matched = False
            for class_feat, weight in class_weights.items():
                # Check if feature names share common prefix (like "No_1", "No_2", etc.)
                if feat.split('_')[0] == class_feat.split('_')[0]:
                    regression_weights.append(weight)
                    matched = True
                    break
            if not matched:
                # Default weight if no match found
                regression_weights.append(1.0 / len(regression_features))

    return np.array(regression_weights)


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
    axes[0, 1].set_title('Weighted Predictions')
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
    plt.savefig(os.path.join(output_dir, 'predictions_comparison.png'), dpi=150)
    print(f"\n  Plot saved to: {os.path.join(output_dir, 'predictions_comparison.png')}")
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
    comparison_df = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'MSE', 'RMSE'],
        'Raw': [metrics_raw['r2'], metrics_raw['mae'], metrics_raw['mse'], metrics_raw['rmse']],
        'Weighted': [metrics_weighted['r2'], metrics_weighted['mae'], metrics_weighted['mse'], metrics_weighted['rmse']]
    })

    # Calculate improvement
    comparison_df['Raw MAE'] = metrics_raw['mae']
    comparison_df['Weighted MAE'] = metrics_weighted['mae']
    comparison_df['MAE Improvement'] = metrics_raw['mae'] - metrics_weighted['mae']

    # Save detailed comparison
    with open(os.path.join(output_dir, 'metrics_comparison.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
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

    print(f"\n  Metrics saved to: {os.path.join(output_dir, 'metrics_comparison.txt')}")


def main():
    """Main evaluation function."""
    print("="*60)
    print("STACKED MODEL EVALUATION")
    print("="*60)

    # Load models
    regression_pipeline, regression_features, _ = load_regression_model()
    classification_model, classification_features, _, _ = load_classification_model()

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

    # Get weighted predictions using classification model weights
    print("\nComputing weighted predictions...")

    # Get regression weights from classification model
    regression_weights = get_regression_weights_from_classification(
        classification_model,
        classification_features,
        regression_features
    )

    print(f"Regression weights (first 10): {regression_weights[:10]}")

    # Use predict_with_feature_scaling with the classification weights
    # Note: We use the raw X_regression_data (not the pipeline) for weighted predictions
    weighted_predictions = predict_with_feature_scaling(
        regression_pipeline.named_steps['model'],  # Get the RandomForestRegressor from pipeline
        X_regression_data.values,
        regression_weights
    )
    print(f"Weighted predictions computed: {len(weighted_predictions)} samples")

    # Evaluate both models
    print("\n" + "-"*40)
    print("EVALUATION")
    print("-"*40)

    metrics_raw = evaluate_model(y_true, raw_predictions, "Raw Predictions")
    metrics_weighted = evaluate_model(y_true, weighted_predictions, "Weighted Predictions")

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
        class_importance_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_feature_importance.csv'), index=False)
        print(f"\n  Classification feature importance saved to: {os.path.join(OUTPUT_DIR, 'classification_feature_importance.csv')}")

    # Get feature importances from the model
    feature_importances = regression_pipeline.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({
        'Feature': regression_features,
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.6f}")

    # Save feature importance to file
    importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)
    print(f"\n  Feature importance saved to: {os.path.join(OUTPUT_DIR, 'feature_importance.csv')}")

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

    # Save predictions to file
    results_df = pd.DataFrame({
        'True_AimoScore': y_true,
        'Raw_Prediction': raw_predictions,
        'Weighted_Prediction': weighted_predictions,
        'Raw_Error': y_true - raw_predictions,
        'Weighted_Error': y_true - weighted_predictions
    })

    results_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)
    print(f"\n  Predictions saved to: {os.path.join(OUTPUT_DIR, 'predictions.csv')}")

    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - predictions.csv (all predictions)")
    print(f"  - metrics_comparison.txt (detailed metrics)")
    print(f"  - predictions_comparison.png (visualizations)")

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
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'evaluation_summary.csv'), index=False)
    print(f"\n  Evaluation summary saved to: {os.path.join(OUTPUT_DIR, 'evaluation_summary.csv')}")

    # Final summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nThe evaluation script successfully:")
    print(f"  1. Loaded the stacked model (regression + classification)")
    print(f"  2. Evaluated on {len(y_true)} test samples")
    print(f"  3. Computed metrics (R², MAE, MSE, RMSE)")
    print(f"  4. Compared raw vs weighted predictions")
    print(f"\nKey findings:")
    print(f"  - Raw predictions achieved R² = {metrics_raw['r2']:.4f}")
    print(f"  - Weighted predictions achieved R² = {metrics_weighted['r2']:.4f}")
    if r2_improvement > 0:
        print(f"  - Weighted predictions improved R² by {r2_improvement:.4f}")
    else:
        print(f"  - Weighted predictions degraded R² by {abs(r2_improvement):.4f}")
    print(f"\nNote: The weighted predictions use feature scaling based on")
    print(f"classification model feature importances. If performance degraded,")
    print(f"this may indicate misalignment between classification and regression")
    print(f"feature priorities, or that the feature scaling approach needs")
    print(f"adjustment (e.g., different weighting strategy).")


if __name__ == "__main__":
    main()
