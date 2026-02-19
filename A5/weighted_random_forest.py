import pandas as pd
import numpy as np
import pickle
import os
from CorrelationFilter import CorrelationFilter


# Get directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Local paths - models loaded from A4/models/ directory
MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models/aimoscores_improved_A4.pkl"
)
CLASSIFICATION_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "../A4/models/weaklink_classifier_rf.pkl",
    #  new classifier without "classes" key "A5/models/weaklink_classifier_rfc_A4.pkl"
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
    global regression_pipe, FEATURE_NAMES, MODEL_METRICS

    if os.path.exists(MODEL_PATH):
        print(f"Loading champion model from {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)

        regression_pipe = artifact["model"]
        FEATURE_NAMES = artifact["feature_columns"]
        MODEL_METRICS = artifact.get("test_metrics", {})

        print(f"Model loaded: {len(FEATURE_NAMES)} features")
        print(f"Test R2: {MODEL_METRICS.get('r2', 'N/A')}")
        return True

    print(f"Champion model not found at {MODEL_PATH}")
    return False


def load_classification_model():
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
        CLASSIFICATION_CLASSES = artifact["classes"]
        CLASSIFICATION_METRICS = artifact.get("test_metrics", {})

        len_features = len(CLASSIFICATION_FEATURE_NAMES)
        print(
            f"Classification model loaded: {len_features} features")
        print(f"Classes: {CLASSIFICATION_CLASSES}")
        return True

    print(f"Classification model not found at {CLASSIFICATION_MODEL_PATH}")
    return False

def load_example():
    if FEATURE_NAMES is None:
        return [0.5] * 35

    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        sample_row = df.sample(1)
        # Return value for each feature
        result = []
        for f in FEATURE_NAMES:
            if f in df.columns:
                val = float(sample_row[f].values[0])
                # Clamp to valid slider range [0, 1]
                val = max(0.0, min(1.0, val))
                result.append(val)
            # using 0.5 as default if feature not in dataset
            else:
                result.append(0.5)
        return result
    except Exception as e:
        print(f"Error loading example: {e}")
        return [0.5] * len(FEATURE_NAMES)


def load_classification_example():
    if CLASSIFICATION_FEATURE_NAMES is None:
        return [0.5] * 40

    try:
        df = pd.read_csv(DATA_PATH, sep=';', decimal=',')
        sample_row = df.sample(1)
        # Return value for each feature
        result = []
        for f in CLASSIFICATION_FEATURE_NAMES:
            if f in df.columns:
                val = float(sample_row[f].values[0])
                # Clamp to valid slider range [0, 1]
                val = max(0.0, min(1.0, val))
                result.append(val)
            # using 0.5 as default if feature not in dataset
            else:
                result.append(0.5)
        return result
    except Exception as e:
        print(f"Error loading classification example: {e}")
        return [0.5] * len(CLASSIFICATION_FEATURE_NAMES)

def predict_with_feature_scaling(rf, X, feature_weights, normalize=True):
    """
    Weighting inputs before passing them to each tree
    and calculating the weighted average of predictions.

    rf: fitted RandomForestRegressor
    X: (n_samples, n_features)
    feature_weights: (n_features,)
    """
    fw = np.asarray(feature_weights, dtype=float)
    if normalize:
        fw = fw / (fw.sum() + 1e-12)
    X_scaled = X * fw  # broadcasting
    # use per-tree predictions and average (same aggregation as rf.predict)
    tree_preds = np.stack([t.predict(X_scaled) for t in rf.estimators_], axis=1)
    return tree_preds.mean(axis=1)


def get_classification_weights(classification_model, X, feature_names):
    """
    Get feature weights from classification model for angle deviation prediction.

    Uses the classification model's feature importances combined with
    class probabilities to determine which features are most important for
    detecting specific deviations.

    classification_model: fitted RandomForestClassifier
    X: input features (n_samples, n_features)
    feature_names: list of feature names

    Returns: dict mapping feature names to weights
    """
    if hasattr(classification_model, 'feature_importances_'):
        # Use feature importances from the model
        importances = classification_model.feature_importances_

        # Get class probabilities for the input
        # Note: X should have the same features as classification model expects
        proba = classification_model.predict_proba(X)[0]

        # Create a combined weight based on:
        # 1. How important each feature is overall (feature_importances_)
        # 2. Which classes are predicted (higher probability classes = more relevant features)

        # Get the top classes by probability
        top_class_indices = np.argsort(proba)[-3:]  # top 3 classes

        # Get feature importances for top classes
        # For RandomForest, we don't have per-class importances directly
        # So we use overall importances weighted by class probability

        # Create feature weights based on overall importance
        weights = dict(zip(feature_names, importances))

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        return weights

    # Fallback: equal weights
    return {f: 1.0/len(feature_names) for f in feature_names}


def get_regression_weights_from_classification(classification_model,
                                               classification_example,
                                               classification_features,
                                               regression_features):
    """
    Map classification model weights to regression feature space.

    classification_model: fitted RandomForestClassifier
    classification_example: input data for classification model
    classification_features: list of feature names for classification
    regression_features: list of feature names for regression

    Returns: numpy array of weights for regression features
    """
    # Get weights from classification model
    class_weights = get_classification_weights(classification_model,
                                               classification_example,
                                               classification_features)

    # Map to regression features
    regression_weights = []
    for feat in regression_features:
        # Try to find matching feature in classification features
        if feat in class_weights:
            regression_weights.append(class_weights[feat])
        else:
            # Check for partial matches (e.g., "LeftKnee" matches "LeftKneeMovesInward")
            matched = False
            for class_feat, weight in class_weights.items():
                if feat in class_feat or class_feat in feat:
                    regression_weights.append(weight)
                    matched = True
                    break
            if not matched:
                # Default weight if no match found
                regression_weights.append(1.0 / len(regression_features))

    return np.array(regression_weights)


def predict_with_classification_weights(regression_model, X,
                                        classification_model,
                                        classification_example,
                                        classification_features,
                                        regression_features):
    """
    Predict using feature scaling based on classification model weights.

    regression_model: fitted RandomForestRegressor
    X: input features for regression (n_samples, n_features)
    classification_model: fitted RandomForestClassifier
    classification_example: input data for classification model
    classification_features: list of feature names for classification
    regression_features: list of feature names for regression

    Returns: predicted scores
    """
    # Get weights from classification model mapped to regression features
    weights = get_regression_weights_from_classification(
        classification_model, classification_example,
        classification_features, regression_features
    )

    print(f"Classification-derived weights: {weights}")

    # Use predict_with_feature_scaling with the classification weights
    return predict_with_feature_scaling(regression_model, X, weights)

if __name__ == "__main__":
    # load the pickled models
    load_champion_model()
    load_classification_model()

    # load examples
    classification_example = load_classification_example()
    #print(len(classification_example)) == 41
    regression_example = load_example()
    #print(len(regression_example)) == 35

    regression_model = regression_pipe[1]
    print(regression_model)

    #classification_model = classification_pipe[1]
    print(classification_model)

    features_df = pd.DataFrame([regression_example], columns=FEATURE_NAMES)
    raw_score = regression_model.predict(features_df)[0]
    print(f"Raw prediction score: {raw_score}")

    # Use classification model weights for feature scaling
    classification_example_df = pd.DataFrame([classification_example], columns=CLASSIFICATION_FEATURE_NAMES)

    # Method 1: Get weights from classification model and use for regression prediction
    weights = get_regression_weights_from_classification(
        classification_model,
        classification_example_df,
        CLASSIFICATION_FEATURE_NAMES,
        FEATURE_NAMES
    )
    print(f"Classification-derived weights for regression: {weights}")

    # Method 2: Use predict_with_classification_weights function
    scaled_prediction = predict_with_classification_weights(
        regression_model,
        features_df,
        classification_model,
        classification_example_df,
        CLASSIFICATION_FEATURE_NAMES,
        FEATURE_NAMES
    )
    print(f"Prediction with classification weights: {scaled_prediction}")

    # Compare with raw prediction
    print(f"Raw prediction: {raw_score}")
    print(f"Difference: {scaled_prediction[0] - raw_score}")
