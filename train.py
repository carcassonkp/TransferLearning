import os
# import cv2
import keras.callbacks
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from sklearn.metrics import confusion_matrix
import random
import cv2

from utils import prepare_dataset
import pathlib

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# --------- DATASET ---------
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMG_SHAPE = 480

X_train, y_train, X_test, y_test = prepare_dataset(data_dir, IMG_SHAPE)
input_shape = X_train.shape[1:]
# --------- Augmentation ---------
AUGMENTATION = True # Disable if augmentation is not wanted
if AUGMENTATION:

    model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=True,
                                                                  weights='imagenet',
                                                                  classes=1000,
                                                                  classifier_activation='softmax',
                                                                  include_preprocessing=False
                                                                  )
    model.trainable = False
    base_output = model.layers[-2].output
    new_output = tf.keras.layers.Dense(5, activation="softmax")(base_output)
    model = tf.keras.models.Model(
    inputs=model.input, outputs=new_output)
    model.summary()
    # Augmentation Layers
    preprocessing_layer = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1. / 255)
        , tf.keras.layers.RandomFlip("horizontal")
        , tf.keras.layers.RandomZoom(0.2)
        , tf.keras.layers.Normalization(
          mean=[0.485, 0.456, 0.406],
          variance=[0.229**2, 0.224**2, 0.225**2],
          axis=3)
         , tf.keras.layers.RandomTranslation(height_factor = 0.05, width_factor=0.05)
    ])
    inputs = keras.Input(shape=input_shape)
    x = preprocessing_layer(inputs)
    outputs = model(x)
    model = keras.Model(inputs, outputs)

    preprocessing_layer.summary()
    model.summary()


else:
    # --------- LOAD MODEL ---------
    model = tf.keras.applications.efficientnet_v2.EfficientNetV2L(include_top=True,
                                                                  weights='imagenet',
                                                                  classes=1000,
                                                                  classifier_activation='softmax',
                                                                  include_preprocessing=True
                                                                  )
    model.trainable = False
    base_output = model.layers[-2].output
    new_output = tf.keras.layers.Dense(5, activation="softmax")(base_output)
    model = tf.keras.models.Model(
    inputs=model.input, outputs=new_output)

LOAD_MODEL = False # disable if it is first time training
if LOAD_MODEL:
    model.load_weights('checkpoint/efficientnetv2_augmentation/')
model.summary()

# --------- MODEL TRAINING ---------
EPOCHS = 50
model_dir = "MODEL_AUGMENTATION/"
checkpoint_dir = os.path.join('checkpoint/', model_dir)
# compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# --------- CALLBACKS ---------
# tensorboard
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
# checkpoint
save_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True, monitor='val_accuracy',
    save_best_only=True,
)
# early stop callback
# earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                      patience=10)
# decreases lr if accuracy does not increase
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10)
# model train
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[save_callback, lr_scheduler, tensorboard_callback
               # , earlyStopCallback
               ])
