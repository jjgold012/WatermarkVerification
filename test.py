# from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from WatermarkVerification1 import *



model_name = 'mnist.w.wm'
net_model, submodel, last_layer_model = utils.splitModel(model_name)
filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))

inputs = np.load('../nn_verification/data/wm.set.npy')
epsilon_max = 0.5
epsilon_interval = 0.001 
for i in range(5):

    input_test = np.reshape(inputs[i], (1,28,28,1))
    
    prediction = np.argmax(net_model.predict(input_test))
    network = Marabou.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
    
    unsat_epsilon, sat_epsilon, sat_vals = findEpsilonInterval(epsilon_max, epsilon_interval, network, prediction)
    epsilonsVars = network.matMulLayers[0]['epsilons']
