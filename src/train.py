# src/train.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def main():
    mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    mlflow.set_tracking_uri(mlflow_uri)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = 10
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = float(accuracy_score(y_test, preds))

    with mlflow.start_run() as run:
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('accuracy', acc)
        os.makedirs('artifacts', exist_ok=True)
        local_model_path = os.path.join('artifacts', 'model.joblib')
        joblib.dump(clf, local_model_path)
        mlflow.log_artifact(local_model_path, artifact_path='local_copy')
        print('Run ID:', run.info.run_id)
        print('Accuracy:', acc)

if __name__ == '__main__':
    main()
