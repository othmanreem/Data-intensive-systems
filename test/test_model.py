import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import re  # For using regular expressions
import matplotlib.pyplot as plt
import datetime

train_path = "../Datasets_all/A2_dataset_80.csv"
test_path = "../Datasets_all/A2_dataset_20.csv"

def save_prediction_plot(y_test, y_test_pred_baseline, baseline_test_r2):
    # Visualize baseline predictions
    fig, axes = plt.subplots(figsize=(5, 5))

    # Actual vs Predicted
    axes.scatter(y_test, y_test_pred_baseline, alpha=0.5)
    axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes.set_xlabel('Actual AimoScore')
    axes.set_ylabel('Predicted AimoScore')
    axes.set_title(f'Baseline: Actual vs Predicted (R²={baseline_test_r2:.4f})')
    axes.grid(True, alpha=0.3)

    # Save the figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # e.g., 20260130_143210
    fig_path = f"baseline_actual_vs_predicted_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

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

    #define y_pred and r2
    y_pred, r2 = 0, 0

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
    else:
        print("no predictions!!!")

    save_prediction_plot(y_test, y_pred, r2)


if __name__ == "__main__":
    model_path = "linear_regression_model.pkl"
    load_and_evaluate_model(model_path)
