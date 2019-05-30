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
    def run(self):
        submodel, last_layer_model = utils.splitModel(net_model)
        filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))

        out_file = open("test.out", "w")
        for i in range(5):

            input_test = np.reshape(self.inputs[i], (1,28,28,1))
            
            prediction = np.argmax(net_model.predict(input_test))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            epsilons_vars = network.matMulLayers[0]['epsilons']
            
            all_vals = sat_vals[1][max(sat_vals[1].keys())][0]
            sat_out = np.array([all_vals[i] for i in range(10)])
            epsilons_vals = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
            
            weights = last_layer_model.get_weights()[0]
            new_weights = weights + epsilons_vals
            
            c = keras.models.clone_model(last_layer_model)
            c.set_weights([new_weights])
            new_out = c.predict(submodel.predict(input_test))[0]
            # print(unsat_epsilon)
            # print(sat_epsilon)
            # print(np.max(epsilons_vals.flatten()))
            # print(np.min(epsilons_vals.flatten()))
            # print(np.average(epsilons_vals.flatten()))
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


inputs = np.load('../nn-verification/data/wm.set.npy')
epsilon_max = 0.5
epsilon_interval = 0.2 
problem = test(net_model, epsilon_max, epsilon_interval, inputs)
problem.run()