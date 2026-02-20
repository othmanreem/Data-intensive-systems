import pandas as pd
import numpy as np
import pickle
import os
import sys
from CorrelationFilter import CorrelationFilter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from A5.mapping import FMS_mapping, NASM_mapping

# Get directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
CLASSIFICATION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../A4/models/gDriveVersion/final_champion_model_A3.pkl"
)
REGRESSION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models/aimoscores_improved_A4.pkl"
)
DATA_PATH = os.path.join(
    SCRIPT_DIR,
    "../A3/A3_Data/train_dataset.csv"
)

regression_pipe = None
FEATURE_NAMES = None
MODEL_METRICS = None

# Classification model
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


def load_champion_model():
    """Load the random forest regression model."""
    global regression_pipe, FEATURE_NAMES, MODEL_METRICS

    if os.path.exists(REGRESSION_MODEL_PATH):
        print(f"Loading champion model from {REGRESSION_MODEL_PATH}")
        with open(REGRESSION_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        regression_pipe = artifact["model"]
        FEATURE_NAMES = artifact["feature_columns"]
        MODEL_METRICS = artifact.get("test_metrics", {})

        print(f"Model loaded: {len(FEATURE_NAMES)} features")
        print(FEATURE_NAMES)
        print(f"Test R2: {MODEL_METRICS.get('r2', 'N/A')}")
        return True

    print(f"Champion model not found at {REGRESSION_MODEL_PATH}")
    return False


def load_classification_model():
    """Load the two-class classification model."""
    global classification_model
    global CLASSIFICATION_FEATURE_NAMES
    global CLASSIFICATION_CLASSES
    global CLASSIFICATION_METRICS

    if os.path.exists(CLASSIFICATION_MODEL_PATH):
        print(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}")
        with open(CLASSIFICATION_MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        classification_model = artifact["model"]
        CLASSIFICATION_FEATURE_NAMES = artifact["feature_columns"]
        CLASSIFICATION_CLASSES = artifact.get("classes", ["lower_body", "upper_body"])
        CLASSIFICATION_METRICS = artifact.get("test_metrics", {})

        print(f"Classification model loaded: {len(CLASSIFICATION_FEATURE_NAMES)} features")
        print(f"Classes: {CLASSIFICATION_CLASSES}")
        return True

    print(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
    return False


def load_example():
    """Load a sample row from training data for testing."""
    if FEATURE_NAMES is None:
        return [0.5] * 35

    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        sample_row = df.sample(1)
        result = []
        for f in FEATURE_NAMES:
            if f in df.columns:
                val = float(sample_row[f].values[0])
                val = max(0.0, min(1.0, val))
                result.append(val)
            else:
                result.append(0.5)
        return result
    except Exception as e:
        print(f"Error loading example: {e}")
        return [0.5] * len(FEATURE_NAMES)


def load_classification_example():
    """Load a sample row from training data for classification testing."""
    if CLASSIFICATION_FEATURE_NAMES is None:
        return [0.5] * 40

    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        sample_row = df.sample(1)
        result = []
        for f in CLASSIFICATION_FEATURE_NAMES:
            if f in df.columns:
                val = float(sample_row[f].values[0])
                val = max(0.0, min(1.0, val))
                result.append(val)
            else:
                result.append(0.5)
        return result
    except Exception as e:
        print(f"Error loading classification example: {e}")
        return [0.5] * len(CLASSIFICATION_FEATURE_NAMES)


def predict_with_feature_scaling(rf, X, feature_weights, normalize=True):
    """
    Weight inputs before passing them to each tree and calculate weighted average.

    rf: fitted RandomForestRegressor
    X: (n_samples, n_features)
    feature_weights: (n_features,)
    """
    fw = np.asarray(feature_weights, dtype=float)
    if normalize:
        fw = fw / (fw.sum() + 1e-12)
    X_scaled = X * fw
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

    print(f"Feature weights: {feature_weights}")

    # Get important body region
    important_region, region_proba = get_important_body_region(classification_model,
                                                                classification_example,
                                                                classification_features)
    print(f"Important body region: {important_region} (probability: {region_proba:.3f})")

    # Use predict_with_feature_scaling with the region-based weights
    prediction = predict_with_feature_scaling(regression_model, X, feature_weights)

    return prediction, feature_weights, important_region


if __name__ == "__main__":
    # Load the pickled models
    load_champion_model()
    load_classification_model()

    if regression_pipe is None or classification_model is None:
        print("Error: Could not load required models")
        exit(1)

    # Load examples
    classification_example = load_classification_example()
    regression_example = load_example()

    regression_model = regression_pipe[1]
    print(f"Regression model: {regression_model}")

    features_df = pd.DataFrame([regression_example], columns=FEATURE_NAMES)
    raw_score = regression_model.predict(features_df)[0]
    print(f"Raw prediction score: {raw_score}")

    # Use classification model weights for feature scaling
    classification_example_df = pd.DataFrame([classification_example], columns=CLASSIFICATION_FEATURE_NAMES)

    # Method: Get body region weights and use for regression prediction
    scaled_prediction, feature_weights, important_region = predict_with_body_region_weights(
        regression_model,
        features_df,
        classification_model,
        classification_example_df,
        CLASSIFICATION_FEATURE_NAMES,
        FEATURE_NAMES
    )

    print(f"Prediction with body region weights: {scaled_prediction}")
    print(f"Important body region: {important_region}")
    print(f"Raw prediction: {raw_score}")
    print(f"Difference: {scaled_prediction[0] - raw_score}")

    # Print recommendation based on important body region
    if important_region == 'upper_body':
        print(f"Recommendation: {BODY_REGION_RECOMMENDATIONS['Upper Body']}")
    else:
        print(f"Recommendation: {BODY_REGION_RECOMMENDATIONS['Lower Body']}")
