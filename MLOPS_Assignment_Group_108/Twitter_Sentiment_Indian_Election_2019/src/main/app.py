import os
import joblib
import mlflow
import pandas as pd
from flask import Flask, request, jsonify
from Twitter_Sentiment_Indian_Election_2019.src.main.train_model_with_grid_search import train_model_with_gs

app = Flask(__name__)

# Environment variables
DATA_FILE_PATH = os.getenv("DATA_FILE_PATH")
PKL_FILE_PATH = os.getenv("PKL_FILE_PATH")

# Global variables
global_df = pd.read_csv(DATA_FILE_PATH)
best_model = None
mlflow_model = None
hyperparam_grid = {
    'tfidf__min_df': [0.001, 0.01],
    'tfidf__max_features': [1000, 1500, 10000],
    'clf__penalty': ['l1', 'l2'],
    'clf__max_iter': [100, 500, 1000]
}


@app.route('/mlops/retrain', methods=['POST'])
def retrain_model():
    global best_model

    try:
        best_model, best_params, evaluation_scores, _ = train_model_with_gs(global_df, hyperparam_grid)
        response = {
            "best_params": best_params,
            "evaluation_scores": evaluation_scores
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 500


@app.route('/mlops/predict', methods=['GET'])
def predict_sentiment():
    try:
        data = request.get_json()
        sentence = data.get('sentence', None)

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        global best_model

        if not best_model:
            try:
                best_model = joblib.load(PKL_FILE_PATH)
            except FileNotFoundError:
                return jsonify({"error": "Model file not found. Retrain using /mlops/retrain"}), 500

        predictions = best_model.predict([sentence])
        prediction_translation = translate_sentiment(predictions[0])

        return jsonify({
            "predicted_class": str(predictions[0]),
            "translation": prediction_translation
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mlops/predict_mlflow', methods=['GET'])
def predict_with_mlflow_model():
    try:
        data = request.get_json()
        sentence = data.get('sentence', None)

        if not sentence:
            return jsonify({"error": "No sentence provided"}), 400

        global mlflow_model

        if not mlflow_model:
            mlflow.set_tracking_uri('http://localhost:5000')
            experiment = mlflow.get_experiment_by_name('Twitter_Sentiment_Analysis')
            latest_run = mlflow.search_runs([experiment.experiment_id]).iloc[0]
            model_uri = f"runs:/{latest_run['run_id']}/best_model"
            mlflow_model = mlflow.sklearn.load_model(model_uri)

        predictions = mlflow_model.predict([sentence])
        prediction_translation = translate_sentiment(predictions[0])

        return jsonify({
            "predicted_class": str(predictions[0]),
            "translation": prediction_translation
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mlops/list_experiments', methods=['GET'])
def list_mlflow_experiments():
    try:
        experiments = mlflow.search_experiments(filter_string='name="Twitter_Sentiment_Analysis"')
        experiment_results = []

        for experiment in experiments:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

            for _, run in runs.iterrows():
                experiment_results.append({
                    "Experiment ID": experiment.experiment_id,
                    "Name": experiment.name,
                    "Run ID": run['run_id'],
                    "Accuracy": run.get('accuracy', 'N/A'),
                    "Precision": run.get('precision', 'N/A'),
                    "Recall": run.get('recall', 'N/A'),
                    "F1 Score": run.get('f1_score', 'N/A'),
                    "Best Params": run.get('best parameters', 'N/A')
                })

        return jsonify({"experiment_results": experiment_results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/mlops/health', methods=['GET'])
def health_check():
    return "Service is running", 200


def translate_sentiment(prediction):
    sentiment_mapping = {
        0: "Neutral Sentiment",
        1: "Positive Sentiment",
        2: "Negative Sentiment"
    }
    return sentiment_mapping.get(prediction, "Unknown Sentiment")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
