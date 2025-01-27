import os
import pandas as pd
import pytest
from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess


@pytest.mark.parametrize("input_text, expected_output", [
    ("Check out https://example.com! It's amazing! #NLP @OpenAI", "check amazing"),
    (12345, ""),  # Assuming non-string input is handled as an empty string
    ("", ""),
    ("the and is of in", ""),
    ("Hello!!! *** $$$ World???", "hello world"),
    ("There are 123 apples and 456 oranges.", "apple orange"),
    ("This Is A Mixed CASE Text.", "mixed case text"),
])
def test_preprocess_function(input_text, expected_output):
    """
    Tests the preprocess function with various inputs to validate its behavior.
    """
    download_nlp()
    assert preprocess(input_text) == expected_output


# Path to the CSV file set via an environment variable
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")


def test_preprocess_on_dataset():
    """
    Tests the preprocess function on a dataset to ensure no null rows are introduced during text cleaning.
    Performs basic data validations and consistency checks.
    """
    # Ensure the environment variable is set
    download_nlp()
    assert DATA_FILE_PATH, "DATA_FILE_PATH environment variable is not set or invalid."

    # Load the dataset
    dataframe = pd.read_csv(DATA_FILE_PATH)

    # Validate that the dataset contains the required column
    assert 'tweet' in dataframe.columns, "'tweet' column not found in the dataset."

    # Apply preprocessing to the 'tweet' column
    dataframe['clean_tweet'] = dataframe['tweet'].apply(preprocess)

    # Check for null values in the processed column
    assert dataframe['clean_tweet'].isnull().sum() == 0, "Null values found in 'clean_tweet' column after preprocessing."
