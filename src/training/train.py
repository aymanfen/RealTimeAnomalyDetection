import mlflow
import mlflow.sklearn
import mlflow.keras
import numpy as np

def trainmodel(model,Xtrain,experimentname='AnomalyDetection',runname=None):
    mlflow.set_experiment(experimentname)
    with mlflow.start_run(run_name=runname):
        model.fit(Xtrain)
        metrics=model.evaluate(Xtrain)

        if hasattr(model,"params"):
            mlflow.log_params(model.params)

        mlflow.log_metrics(metrics)

        if model.framework=='sklearn':
            mlflow.sklearn.log_model(
                model.model,
                artifact_path="model"
            )
            
        elif model.framework=='keras':
            mlflow.keras.log_model(
                model.model,
                artifact_path="model"
            )

        return metrics