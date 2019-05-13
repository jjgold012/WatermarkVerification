# from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from WatermarkVerification1 import *
from copy import deepcopy
from tensorflow import keras

model_name = 'mnist.w.wm'
net_model, submodel, last_layer_model = utils.splitModel(model_name)
filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))

inputs = np.load('../nn_verification/data/wm.set.npy')
epsilon_max = 0.5
epsilon_interval = 0.2 
out_file = open("test.out", "w")
for i in range(5):

    input_test = np.reshape(inputs[i], (1,28,28,1))
    
    prediction = np.argmax(net_model.predict(input_test))
    network = Marabou.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
    
    unsat_epsilon, sat_epsilon, sat_vals = findEpsilonInterval(epsilon_max, epsilon_interval, network, prediction)
    epsilons_vars = network.matMulLayers[0]['epsilons']
    all_vals = sat_vals[1][max(sat_vals[1].keys())][0]
    sat_out = [all_vals[i] for i in range(10)]
    epsilons_vals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
    weights = last_layer_model.get_weights()[0]
    new_weights = weights + epsilons_vals
    c = keras.models.clone_model(last_layer_model)
    c.set_weights([new_weights])
    out_file.write('{}\n'.format(np.array(sat_out)))
    out_file.write('{}\n'.format(c.predict(submodel.predict(input_test))[0]))
out_file.close()

