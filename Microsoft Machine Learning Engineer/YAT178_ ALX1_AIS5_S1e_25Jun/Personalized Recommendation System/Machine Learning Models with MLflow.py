import os
import mlflow
from mlflow_utils import create_mlflow_experiment, get_mlflow_experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


def log_training_progress(model, X, y):
    """Log training progress metrics."""
    model.fit(X, y)
    return model

if __name__ == "__main__":
    # Try to get the existing experiment or create a new one if it doesn't exist
    experiment = get_mlflow_experiment(experiment_name="feedback classification")
    
    if experiment is None:
        print("Experiment not found. Creating a new experiment.")
        experiment_id = mlflow.create_experiment("feedback classification")
    else:
        experiment_id = experiment.experiment_id

    # Create the metrics directory if it doesn't exist
    if not os.path.exists("metrics"):
        os.makedirs("metrics")

    with mlflow.start_run(run_name="run", experiment_id=experiment_id) as run:
        # Load and preprocess data
        balanced_df = pd.read_csv("balanced_df.csv")
        balanced_df.dropna(inplace=True)
        X = balanced_df['review']
        Y = balanced_df['review-label'] - 1  # Ensure this shift is appropriate for your label values

        vectorizer = TfidfVectorizer()
        vectorizer.fit(X)
        X = vectorizer.transform(X)

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)

        # Convert sparse matrix to NumPy arrays
        x_train_np = x_train.toarray()  
        x_test_np = x_test.toarray()    
        y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

        mlflow.autolog()  # Automatically log parameters, metrics, and models

        # Initialize the VotingClassifier
        model = VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=500, verbose=0)),
            ('xgb', XGBClassifier()),
        ], voting='soft')

        # Log training metrics
        print("Fitting the model...")
        model = log_training_progress(model, x_train_np, y_train_np)

        # Make predictions
        train_pred = model.predict(x_train_np)
        test_pred = model.predict(x_test_np)

        # Log metrics using MLflow
        mlflow.log_metric("train_accuracy", np.mean(train_pred == y_train_np))
        mlflow.log_metric("test_accuracy", np.mean(test_pred == y_test))

        # Save and log classification report
        print("Saving classification report...")
        classification_report_str = classification_report(y_test, test_pred)
        with open("metrics/classification_metrics.txt", "w") as f:
            f.write(classification_report_str)

        mlflow.log_artifact("metrics/classification_metrics.txt")

        # Log confusion matrix
        print("Logging confusion matrix...")
        fig_conf_matrix = plt.figure()
        ConfusionMatrixDisplay.from_predictions(y_test, test_pred, ax=plt.gca())
        plt.title("Confusion Matrix")
        mlflow.log_figure(fig_conf_matrix, "metrics/confusion_matrix.png")
        
        # Register the model
        model_name = "VotingClassifierModel"
        mlflow.sklearn.log_model(model, model_name)
        
        # Optionally register the model
        mlflow.register_model("runs:/{}/{}".format(run.info.run_id, model_name), model_name)

        print("Run completed and logged in MLflow.")
