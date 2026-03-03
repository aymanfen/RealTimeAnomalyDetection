
def ifsearchspace(trial):
    return {
        "nestimators" : trial.suggest_int("nestimators",100,400),
        "maxsamples" : trial.suggest_float("maxsamples",0.5,1.0),
        "contamination" : trial.suggest_float("contamination",0.01,0.1)
    }

def aesearchspace(trial):
    return {
        "input_dim":INPUT_DIM,
        "latent_dim":trial.suggest_int("latent_dim",4,32),
        "lr":trial.suggest_float("lr",1e-4,1e-2,log=True),
        "epochs":trial.suggest_int("epochs",20,100),
        "batch_size":trial.suggest_categorical([64,128,256])
    }

def somsearchspace(trial):
    return {
        'x':trial.suggest_int("x",5,20),
        'y':trial.suggest_int("y",5,20),
        "sigma":trial.suggest_float("sigma",0.5,2.0),
        "learning_rate":trial.suggest_float("learning_rate",0.1,0.8),
        "iterations":trial.suggest_int("iterations",500,3000)
    }