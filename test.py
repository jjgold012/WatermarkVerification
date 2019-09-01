import numpy as np
import os
import argparse
import utils
from copy import deepcopy
import MarabouNetworkTFWeightsAsVar
from maraboupy import MarabouUtils
from WatermarkVerification1 import *
from copy import deepcopy
from tensorflow import keras
from pprint import pprint

class test(WatermarkVerification):
    def run(self, model_name):

        net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
        submodel, last_layer_model = utils.splitModel(net_model)
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)

        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        out_file = open("test.out", "w")
        for i in range(5):

            
            prediction = np.argmax(predictions[i])
            inputVals = np.reshape(lastlayer_inputs[i], (1, lastlayer_inputs[i].shape[0]))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=inputVals)
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            epsilons_vars = network.matMulLayers[0]['epsilons']
            
            all_vals = sat_vals[1][0]
            sat_out = np.array([all_vals[i] for i in range(10)])
            epsilons_vals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
            
            weights = last_layer_model.get_weights()[0]
            new_weights = weights + epsilons_vals
            
            c = keras.models.clone_model(last_layer_model)
            c.set_weights([new_weights])
            new_out = c.predict(inputVals)[0]
            out_file.write('Marabou out:\n')
            pprint(sat_out.tolist(), out_file)
            out_file.write('Network out:\n')
            pprint(new_out.tolist(), out_file)
            out_file.write('Diff:\n')
            pprint((sat_out - new_out).tolist(), out_file)
            out_file.write('\n')
            
        out_file.close()



model_name = 'mnist.w.wm'
MODELS_PATH = './Models'
net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))


inputs = np.load('./data/wm.set.npy')
# a = np.max(np.max(inputs, axis=2) , axis=1)
epsilon_max = 0.5
epsilon_interval = 0.2 
problem = test(epsilon_max, epsilon_interval)
problem.run(model_name)