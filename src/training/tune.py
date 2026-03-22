import optuna
import mlflow
from src.training.train import trainmodel


def tunemodel(modelclass,modelsearchspace,Xtrain,target,dir,ntrials=20,experimentname="AnomalyTuning"):
    mlflow.set_experiment(experiment_name=experimentname)

    def objective(trial):
        params=modelsearchspace(trial)
        model=modelclass(**params)

        with mlflow.start_run(nested=True,run_name=f"trial{trial.number}"):
            model.fit(Xtrain)
            metrics=model.evaluate(Xtrain)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            #optimization target
            return metrics[target]
        
    study=optuna.create_study(direction=dir)
    study.optimize(objective,n_trials=ntrials)
    return study
