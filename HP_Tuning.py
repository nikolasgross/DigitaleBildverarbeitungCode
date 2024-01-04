# %% IMPORT LIBRARIES
import checker
import generator

from IPython.display import display, clear_output
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import importlib
importlib.reload(checker)
importlib.reload(generator)

import tensorflow as tf 
from keras.utils import to_categorical
from sklearn.decomposition import PCA
import tensorflow.keras as keras
import keras_tuner 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import os
cwd = os.getcwd ()

# %% Loading the MNIST dataset in one line
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# %% SCALE DATA AND RESHAPE FOR NN
x_train_normalized = x_train/255
x_test_normalized = x_test/255

x_train= x_train_normalized.reshape(-1, 784)
x_test = x_test_normalized.reshape(-1, 784)
#%% PCA
pca = PCA(n_components = 0.9)
x_train_reduced = pca.fit_transform(x_train)
x_test_reduced = pca.transform(x_test)

print("Number of inputs after PCA:",pca.n_components_)
print('x_red: x_train after PCA', x_train_reduced.shape)
#%%
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):

        nr_hid_layers=hp.Int(name="nr_hid_layers",min_value=1, max_value=3,step=1)    
        nr_neurons=hp.Int(name="nr_neurons",min_value=2, max_value=64,step=2, sampling="log")      
        dropout=hp.Boolean("dropout")
        dropout_rate=hp.Float(name="dropout_rate",min_value=0.1,max_value=0.4,step=0.1)
        weight_init=hp.Choice("weight_init", values =["glorot_uniform","glorot_normal"])
        activation_function=hp.Choice("activation_func",values=["tanh","sigmoid"])

        marvin=tf.keras.models.Sequential()
        marvin.add(tf.keras.layers.Flatten(input_shape=(pca.n_components_,)))
        for _ in range (nr_hid_layers):
            marvin.add(tf.keras.layers.Dense(nr_neurons, activation=activation_function,kernel_initializer=weight_init))
           
        if dropout:
            marvin.add(tf.keras.layers.Dropout(rate=dropout_rate))
        
        marvin.add(tf.keras.layers.Dense(10, activation='softmax',kernel_initializer=weight_init))
        
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        print(-tf.math.log(1/10))
        sampleID = 100
        loss_fn(y_train[:1], marvin(x_train_reduced[sampleID-1:sampleID]).numpy()).numpy()

        marvin.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_fn, metrics=['accuracy'])

        return marvin

        
    def fit(self, hp, marvin, *args, **kwargs):
        marvin.summary()
        return marvin.fit(*args,batch_size=32,**kwargs)
       

#%% SEARCH ALGORITHM 
tuner = keras_tuner.RandomSearch(
    hypermodel=MyHyperModel(),
    objective="val_accuracy",
    max_trials=500,
    executions_per_trial=1,
    directory= os.path.join(cwd,'hp_study/random_search'),
    project_name="hp_ffnn_mnist",
)

#%%
# %% CALLBACKS
early_stopping = EarlyStopping (monitor='val_accuracy',patience=6,mode="max") 
tensorboard = TensorBoard (log_dir = os.path.join (cwd ,'hp_study/tensorboard_logs'))
reduce_lr = ReduceLROnPlateau (monitor ='val_accuracy', factor =0.5 , min_delta=0.00005 , patience=4 , min_lr=0.0001, mode="max")

# %% SEARCH FOR BEST HPs
if __name__ == '__main__':
    tuner.search(x_train_reduced,y_train,epochs=50,validation_data=(x_test_reduced,y_test),callbacks =[early_stopping,tensorboard,reduce_lr])
