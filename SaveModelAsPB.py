from nn_verification.utils import load_model
import numpy as np
import os
import uuid

import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from MarabouNetworkAsVarTF import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=' Construct Siamese Network.')
    parser.add_argument('--model', help='the name of the model')
    args = parser.parse_args()

    MODELS_PATH = './Models'
    model_name = args.model
    net_model = load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    print('Model Summary')
    print(net_model.summary())

    submodel = keras.Model(inputs=net_model.inputs, outputs=net_model.layers[-2].output, name='submodel-'+str(uuid.uuid4())[:5])
    print('SubModel Summary')
    print(submodel.summary())

    input_test = np.reshape((np.array(range(16))/float(10)), (1,4,4,1))
    print(input_test)
    print(submodel.predict(input_test))

    in_shape = (submodel.output.shape[1].value,)
    print(in_shape)
    last_layer_in = keras.layers.Input(shape=in_shape, name='ll_input-'+str(uuid.uuid4())[:5])
    last_layer_model = keras.Model(inputs=last_layer_in, outputs=net_model.layers[-1](last_layer_in), name='lastlayer-'+str(uuid.uuid4())[:5])
    print('Last Layer Summary')
    print(last_layer_model.summary())

    print(last_layer_model.predict(submodel.predict(input_test)))
    print(net_model.predict(input_test))

    