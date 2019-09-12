import numpy as np
import tensorflow as tf
import os
import argparse
import utils
from copy import deepcopy
# import MarabouNetworkTFWeightsAsVar
# from maraboupy import MarabouUtils
# from WatermarkVerification1 import *
from copy import deepcopy
from tensorflow import keras
from pprint import pprint

model_name = 'mnist.w.wm'
MODELS_PATH = './Models'
net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

for i in [1,2,3,4,5,6,7,25,50,75,100]:
    out_file = open('./data/results/problem3/{}.{}.wm.accuracy.csv'.format(model_name, i), 'w')
    out_file.write('test-accuracy,test-loss,train-accuracy,train-loss\n')
    out_file.flush()
    epsilons = np.load('./data/results/problem3/{}.{}.wm.vals.npy'.format(model_name, i))
    for j in range(epsilons.shape[0]):
        weights = net_model.get_weights()
        weights[-1] = weights[-1] + epsilons[j]

        copy_model = keras.models.clone_model(net_model)
        copy_model.set_weights(weights)
        copy_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        test_loss, test_acc = copy_model.evaluate(x_test, y_test)
        train_loss, train_acc = copy_model.evaluate(x_train, y_train)
        out_file.write('{},{},{},{}\n'.format(test_acc, test_loss, train_acc, train_loss))
        out_file.flush()

    out_file.close()
    