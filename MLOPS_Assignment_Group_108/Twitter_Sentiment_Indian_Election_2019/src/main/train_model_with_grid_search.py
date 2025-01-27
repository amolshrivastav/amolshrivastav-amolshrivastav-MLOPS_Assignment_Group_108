import os
import time
import joblib
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from Twitter_Sentiment_Indian_Election_2019.src.main.download_nlp_dependencies import download_nlp
from Twitter_Sentiment_Indian_Election_2019.src.main.preprocess_text import preprocess

# Environment variable for pickle file path
PKL_FILE_PATH = os.getenv("PKL_FILE_PATH")


def train_model_with_grid_search(dataframe, hyperparameter_grid):
    """
    Trains a machine learning model using TF-IDF and Logistic Regression with GridSearchCV.
    Logs the results and metrics using MLflow and saves the best model to a file.
    """
    # Download necessary NLP resources
    download_nlp()

    # Preprocess the tweets in the dataframe
    dataframe['cleaner_tweet'] = dataframe['tweet'].apply(preprocess)

    # Drop rows with missing values
    dataframe = dataframe.dropna(subset=['category', 'cleaner_tweet'])
    number_of_records = len(dataframe)

    # Split the data into train and test sets
    documents = dataframe['cleaner_tweet']
    labels = dataframe['category']
    x_train, x_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

    # Define the ML pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # TF-IDF Vectorizer
        ('clf', LogisticRegression())  # Logistic Regression classifier
    ])

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(pipeline, hyperparameter_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # Configure MLflow experiment
    mlflow.set_tracking_uri('http://localhost:5000')
    if not mlflow.get_experiment_by_name('Twitter_Sentiment_Analysis'):
        mlflow.create_experiment('Twitter_Sentiment_Analysis')
    mlflow.set_experiment('Twitter_Sentiment_Analysis')

    # Start MLflow run
    start_time = time.time()
    with mlflow.start_run():
        mlflow.set_tag("Dataset Size", f"{number_of_records} Tweets")
        grid_search.fit(documents, labels)

        # Get the best model and calculate performance metrics
        best_model = grid_search.best_estimator_
        y_predictions = best_model.predict(x_test)
        elapsed_time = time.time() - start_time

        model_scores = calculate_performance_metrics(y_test, y_predictions, elapsed_time)

        # Log the model and metrics to MLflow
        mlflow.sklearn.log_model(best_model, "best_model")
        mlflow.log_metric('accuracy', model_scores["accuracy"])
        mlflow.log_metric('precision', model_scores["precision"])
        mlflow.log_metric('recall', model_scores["recall"])
        mlflow.log_metric('f1_score', model_scores["f1_score"])
        mlflow.log_param("best parameters", grid_search.best_params_)

    # Save the best model to a pickle file
    joblib.dump(best_model, PKL_FILE_PATH)

    print(f"Best parameters found: {grid_search.best_params_}")
    return best_model, grid_search.best_params_, model_scores, grid_search.cv_results_


def calculate_performance_metrics(y_true, y_pred, elapsed_time):
    """
    Calculates performance metrics (accuracy, precision, recall, F1 score) and time taken for predictions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "time_taken": f"{elapsed_time:.2f} seconds"
    }
