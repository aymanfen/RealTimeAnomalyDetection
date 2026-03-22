import mlflow.pyfunc
import pickle

class AnomalyModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import pickle

        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input):
        return self.model.score(model_input)