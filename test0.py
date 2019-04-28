import tensorflow as tf
from tensorflow import keras
import sys
sys.path.insert(0,'..')
from nn_verification.utils import save_model, load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import uuid
from time import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

sess = tf.keras.backend.get_session()

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# params
MODELS_PATH = '../Models'

# ======== build the model ======== #
in1 = keras.layers.Input(shape=(4,4,1), name='input-'+str(uuid.uuid4())[:5])
flat = keras.layers.Flatten(name='flatten-'+str(uuid.uuid4())[:5])(in1)
hidden = keras.layers.Dense(3, activation='relu', name='dense-'+str(uuid.uuid4())[:5])(flat)
output_name = 'output-'+str(uuid.uuid4())[:5]
out1 = keras.layers.Dense(1, activation='relu', name='output-'+str(uuid.uuid4())[:5])(hidden)
model =  keras.Model(inputs=in1, outputs=out1, name='model-'+str(uuid.uuid4())[:5])

print(model.summary())
file_writer = tf.summary.FileWriter('./logs', sess.graph)

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=float(0.001)),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
test_images = np.zeros((2,4,4))
test_images[1] = np.ones((4,4))
test_labels = np.array([0.99,0])
model.fit(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],1 ), test_labels, verbose=1, callbacks=[tensorboard])
test_loss, test_acc = model.evaluate(test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2],1 ), test_labels, verbose=10)
input_test = np.reshape(np.array(range(16)), (1,4,4,1))
print('Test accuracy: {0}, Test loss: {1}'.format(test_acc, test_loss))
if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)
save_model(os.path.join(MODELS_PATH, 'test0_model.json'), os.path.join(MODELS_PATH, 'test0_model.h5'), model)

print("\ntest input")
# print(input_test)
print(model.predict(input_test))
# print(model.summary())
# model.layers.pop()


