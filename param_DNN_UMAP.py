import keras 
import tensorflow as tf        
import umap
import datetime
import os
import numpy as np
cwd = os.getcwd ()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%
x_train_normalized = x_train/255
x_test_normalized = x_test/255
x_train, y_train = x_train_normalized, y_train

x_train= x_train.reshape(-1, 784)
x_test = x_test_normalized.reshape(-1, 784)
#%%

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_test:', x_test.shape)
print('y_test:', y_test.shape)

n_components=1
umap_model = umap.UMAP(n_components=n_components,)
x_train_reduced = umap_model.fit_transform(x_train)
x_test_reduced = umap_model.transform(x_test)

print('x_red:', x_train_reduced.shape)
#%%
marvin = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(n_components,)),
  tf.keras.layers.Dropout(0.1),#
  tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='glorot_uniform', use_bias=True)
])
marvin.summary()
#%% 
marvin.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

#%%
early_stopping = tf.keras.callbacks.EarlyStopping (monitor='val_accuracy',patience=15,mode="max") 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau (monitor ='val_accuracy', factor =0.5 , min_delta=0.00005 , 
                                                  patience=6, min_lr=0.0001, mode="max")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint (filepath=os.path.join(cwd,'models/param_UMAP.h5'),
                                                       monitor='val_accuracy',save_best_only=True, mode="max")
#%%

marvin.fit(
    x_train_reduced,
    y_train,
    epochs= 50,
    batch_size= 64,
    validation_data=(x_test_reduced, y_test),
    callbacks=[model_checkpoint,reduce_lr]
)


#%% SAVE MODEL 
model_name = 'param_UMAP'
marvin.save(model_name, save_format='h5')

print('Success! You saved Marvin as: ', model_name)
#%% EVALUATE MODEL (just to be sure the last val acc value was right)
marvin_reloaded = tf.keras.models.load_model('models/param_UMAP.h5')
marvin_reloaded.summary()
loss_and_metrics = marvin_reloaded.evaluate(x_test_reduced, y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

#%%

# Let Marvin predict on the test set, so we have some data to evaluate his performance.
predictions = marvin_reloaded.predict([x_test_reduced])

# Remember that the prediction of Marvin is a probability distribution over all ten-digit classes
# We want him to assign the digit class with the highest probability to the sample.
predictions = np.argmax(predictions, axis=1)
#pd.DataFrame(predictions)
