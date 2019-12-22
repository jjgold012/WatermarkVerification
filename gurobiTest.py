import numpy as np
import tensorflow as tf
import os
import utils
from tensorflow import keras
from pprint import pprint

model_name = 'mnist.w.wm'
MODELS_PATH = './Models'

epsilons = np.load('./data/results/problem4/{}.2.wm_0-1.vals.npy'.format(model_name))
randomSamples = np.load('./data/random/2.wm.1000.random_samples.npy'.format(model_name))
wm_images = np.load('./data/wm.set.npy')
wm_images = wm_images.reshape(wm_images.shape[0], wm_images.shape[1], wm_images.shape[2],1)

for j in range(epsilons.shape[0]):
    net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))

    weights = net_model.get_weights()
    weights[-1] = weights[-1] + epsilons[j]

    net_model.set_weights(weights)
    net_model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    predictions = np.load('./data/{}.prediction.npy'.format(model_name))
    predIndices = np.flip(np.argsort(predictions, axis=1), axis=1)        
    a = predIndices[:,0] 
    b = predIndices[:,1]
    p = net_model.predict(wm_images)
    [print(p[i]) for i in randomSamples[j]]
    c = np.array([p[i][a[i]] - p[i][b[i]] for i in randomSamples[j]])
    print(c)
