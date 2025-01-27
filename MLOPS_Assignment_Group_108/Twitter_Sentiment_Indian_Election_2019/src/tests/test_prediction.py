import os
import joblib

# Environment variables for file paths
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
PKL_FILE_PATH = os.getenv("PKL_FILE_PATH")


def test_predict_positive_class():
    """
    Tests the model's ability to correctly classify a positive sentiment.
    """
    # Load the trained model
    best_model = joblib.load(PKL_FILE_PATH)

    # Input text for testing
    test_sentence = "This election is going to make a positive impact to people"

    # Get predictions
    predictions = best_model.predict([test_sentence])

    # Assert that the prediction is positive
    assert predictions[0] == 1, f"Expected sentiment to be 1 (positive), but got {predictions[0]}"


def test_predict_negative_class():
    """
    Tests the model's ability to correctly classify a negative sentiment.
    """
    # Load the trained model
    best_model = joblib.load(PKL_FILE_PATH)

    # Input text for testing
    test_sentence = "Political parties are all corrupt and incompetent."

    # Get predictions
    predictions = best_model.predict([test_sentence])

    # Assert that the prediction is negative
    assert predictions[0] == -1, f"Expected sentiment to be -1 (negative), but got {predictions[0]}"


def test_predict_neutral_class():
    """
    Tests the model's ability to correctly classify a neutral sentiment.
    """
    # Load the trained model
    best_model = joblib.load(PKL_FILE_PATH)

    # Input text for testing
    test_sentence = "I have no opinion on this election"

    # Get predictions
    predictions = best_model.predict([test_sentence])

    # Assert that the prediction is neutral
    assert predictions[0] == 0, f"Expected sentiment to be 0 (neutral), but got {predictions[0]}"
