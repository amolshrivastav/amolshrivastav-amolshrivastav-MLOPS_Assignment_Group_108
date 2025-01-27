import os
import numpy as np
import pandas as pd
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model_with_grid_search import train_model_with_gs


def test_train_model_with_grid_search():
    """
    Tests the train_model_with_gs function to ensure it works correctly with a simplified parameter grid.
    Validates the performance of the model by checking evaluation scores and cross-validation scores.
    """
    # Get the dataset path from the environment variable
    data_file_path = os.getenv("DATA_FILE_PATH")
    assert data_file_path, "DATA_FILE_PATH environment variable is not set or invalid."

    # Load the dataset
    dataframe = pd.read_csv(data_file_path)

    # Define a simplified hyperparameter grid
    hyperparameter_grid = {
        'tfidf__min_df': [0.001],           # Minimum document frequency for TF-IDF
        'tfidf__max_features': [1000],     # Maximum number of features for TF-IDF
        'clf__penalty': ['l1'],            # L1 regularization
        'clf__solver': ['liblinear'],      # Solver for Logistic Regression
        'clf__max_iter': [1000]            # Maximum number of iterations for Logistic Regression
    }

    # Train the model and get results
    best_model, best_params, evaluation_scores, cv_scores = train_model_with_gs(dataframe, hyperparameter_grid)

    # Assertions to validate model performance
    mean_test_score = cv_scores['mean_test_score']
    assert mean_test_score > 0.50, f"Expected mean CV accuracy > 0.50, but got {mean_test_score}"
    assert np.all(mean_test_score > 0.50), f"Some CV scores are below 0.50: {cv_scores}"
    assert evaluation_scores['accuracy'] > 0.50, f"Expected accuracy > 0.50, but got {evaluation_scores['accuracy']}"
    assert evaluation_scores['precision'] > 0.50, f"Expected precision > 0.50, but got {evaluation_scores['precision']}"
    assert evaluation_scores['recall'] > 0.50, f"Expected recall > 0.50, but got {evaluation_scores['recall']}"
    assert evaluation_scores['f1_score'] > 0.50, f"Expected F1 score > 0.50, but got {evaluation_scores['f1_score']}"
