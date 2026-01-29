import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

train_path = "../Datasets_all/A2_dataset_80.csv"
test_path = "../Datasets_all/A2_dataset_20.csv"

# validating the linear regression model based on
# https://medium.com/@_SSP/validating-machine-learning-regression-models-a-comprehensive-guide-b94fd94e339c


def load_and_evaluate_model(model_path):
    # Load the pickled model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # check the model type
    assert isinstance(model, LinearRegression)

    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Define target and features
    target_col = "AimoScore"
    unwanted_cols = ["EstimatedScore"]
    features_cols = [
        col
        for col in train_df.columns
        if col not in unwanted_cols and col != target_col
    ]

    X_test = test_df[features_cols]
    y_test = test_df[target_col]

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error on test set: {mae:.4f}")
    print(f"R^2 score on test set: {r2:.4f}")

    # assert the threeshold values
    assert mae < 0.15, "Mean Absolute Error is too high"
    assert r2 > 0.5, "R^2 score is too low"

    # Save predictions to CSV
    test_df["Predicted_AimoScore"] = y_pred
    test_df.to_csv("predicted_test.csv", index=False)


if __name__ == "__main__":
    model_path = "linear_regression_model.pkl"
    load_and_evaluate_model(model_path)
