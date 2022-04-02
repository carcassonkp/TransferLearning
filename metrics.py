import os
# import cv2
import keras.callbacks
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix,f1_score
from utils import plot_confusion_matrix, prepare_dataset
from sklearn.metrics import classification_report
import pathlib

# --------- LOAD DATASET ---------
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMG_SHAPE = 480

X_train, y_train, X_test, y_test = prepare_dataset(data_dir, IMG_SHAPE)
input_shape = X_train.shape[1:]
# --------- LOAD MODEL ---------
AUGMENTATION = True
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
        # , tf.keras.layers.Normalization(
        #   mean=[0.485, 0.456, 0.406],
        #   variance=[0.229**2, 0.224**2, 0.225**2],
        #   axis=3)
        #  , tf.keras.layers.RandomTranslation(height_factor = 0.05, width_factor=0.05)
    ])
    inputs = keras.Input(shape=input_shape)
    x = preprocessing_layer(inputs)
    outputs = model(x)
    model = keras.Model(inputs, outputs)
else:
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



model.load_weights('checkpoint/MODEL_AUGMENTATION/')  # load pretrained model
model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# --------- MODEL EVALUATION ---------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

predictions = model.predict(X_test)
print("Accuracy on test set is:{}".format(test_acc))
print("Loss on test set is:{}".format(test_loss))
rounded_predictions = np.argmax(predictions, axis=-1)
cm = confusion_matrix(y_true=y_test, y_pred=rounded_predictions)
plot_confusion_matrix(cm=cm, classes=classes, title='Confusion Matrix')
res = tf.math.confusion_matrix(y_test, rounded_predictions)
print("Confusion Matrix", res)

print("Classification report: \n", (classification_report(y_test, rounded_predictions)))