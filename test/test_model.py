import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import re  # For using regular expressions

train_path = "../Datasets_all/A2_dataset_80.csv"
test_path = "../Datasets_all/A2_dataset_20.csv"

def extract_missing_feature(error_message):
    # Use regex to find feature names in the ValueError message
    match = re.search(r"Feature names unseen at fit time:\s*-\s*(.+)", error_message)
    if match:
        return match.group(1).strip().split(', ')  # Return list of feature names
    return []

def load_and_evaluate_model(model_path):
    # Load the pickled model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Check the model type
    assert isinstance(model, LinearRegression)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Define target and features dynamically
    target_col = "AimoScore"
    unwanted_cols = ["EstimatedScore"]

    features_cols = [
        col for col in train_df.columns
        if col not in unwanted_cols and col != target_col
    ]

    # Initialize features for prediction
    X_test = test_df[features_cols]
    y_test = test_df[target_col]

    #define y_pred
    y_pred = 0

    # Continue to predict until no ValueErrors occur
    while True:
        try:
            # Predict on test set
            y_pred = model.predict(X_test)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Mean Absolute Error on test set: {mae:.4f}")
            print(f"R^2 score on test set: {r2:.4f}")

            # Assert the threshold values
            assert mae < 0.15, "Mean Absolute Error is too high"
            assert r2 > 0.5, "R^2 score is too low"
            break  # Exit the loop if no errors occur

        except ValueError as e:
            print(f"Error during prediction: {e}")

            # Extract missing features from the error message
            missing_features = extract_missing_feature(str(e))

            # Remove missing features from X_test and features_cols
            if missing_features:
                print(f"Removing missing features from test set: {missing_features}")
                # Update features list
                features_cols = [col for col in features_cols if col not in missing_features]
                # Update X_test
                X_test = X_test[features_cols]
            else:
                print("No more features can be removed, stopping execution.")
                break  # Exit if there are no more features to remove

    if 'y_pred' in locals():  # Check if predictions were made
        # Save predictions to CSV
        test_df["Predicted_AimoScore"] = y_pred
        test_df.to_csv("predicted_test.csv", index=False)


if __name__ == "__main__":
    model_path = "linear_regression_model.pkl"
    load_and_evaluate_model(model_path)
