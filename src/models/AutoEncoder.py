from src.models.BaseClass import BaseAnomalyModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def AutoEncoder(inputdim,latentdim,lr ):
    # Input layers
    inputs = Input(shape=(inputdim,))

    # Encoder
    x = Dense(12, activation="relu")(inputs)
    x = Dense(8, activation="relu")(x)
    latent = Dense(latentdim, activation="relu")(x)

    # Decoder
    x = Dense(8, activation="relu")(latent)
    x = Dense(12, activation="relu")(x)
    output = Dense(inputdim, activation="linear")(x)

    # Autoencoder model
    autoencoder = Model(inputs=inputs, outputs=output)
    autoencoder.compile(optimizer=Adam(learning_rate=lr), loss="mse")

    return autoencoder

class AutoEncoderModel(BaseAnomalyModel):
    def __init__(self,
                 inputdim=18,latentdim=6,lr=0.001,
                 epochs=10,batchsize=64,validationsplit=0.1):
        
        self.params={
            "inputdim":inputdim,
            "latentdim":latentdim,
            "lr":lr,
            "epochs":epochs,
            "batchsize":batchsize,
            "validationsplit":validationsplit
        }
        self.model=AutoEncoder(inputdim,latentdim,lr,)
        self.epochs=epochs
        self.batch_size=batchsize
        self.validation_split=0.1
        
          
    def fit(self,X,**fit_kwargs ):
        # if not isinstance(X, np.ndarray):
        #     X = X.to_numpy()

        callbacks=fit_kwargs.get("callbacks",[])
        earlystop=EarlyStopping(monitor='val_loss',patience=10,
                                restore_best_weights=True,verbose=0)
        callbacks.append(earlystop)

        history=self.model.fit(X,X,
                       epochs=self.epochs,batch_size=self.batch_size,
                       validation_split=self.validation_split,callbacks=callbacks,
                       shuffle=True)
        return history

    def reconstructionerror(self,X):
        # if not isinstance(X, np.ndarray):
        #     X = X.to_numpy()

        recon=self.model.predict(X)
        return np.mean((X-recon)**2,axis=1)


    def score(self,X):
        return self.reconstructionerror(X)