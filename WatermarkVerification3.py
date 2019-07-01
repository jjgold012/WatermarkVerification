import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from tensorflow import keras

from WatermarkVerification1 import *
import MarabouNetworkTFWeightsAsVar2
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkVerification3(WatermarkVerification):


    def run(self, model_name):        
        
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        out_file = open("WatermarkVerification3.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        out_file.flush()
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        network = MarabouNetworkTFWeightsAsVar2.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)

        for i in range(network.epsilons.shape[0]):
            for j in range(network.epsilons.shape[1]):
                network.setUpperBound(network.epsilons[i][j], 1)
                network.setLowerBound(network.epsilons[i][j], 1)
        vals = network.solve()
        print(predictions)

        epsilons_vars = network.epsilons
            
        epsilons_vals = np.array([[vals[0][epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
        
        net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
        submodel, last_layer_model = utils.splitModel(net_model)
        weights = last_layer_model.get_weights()[0]
        new_weights = weights + epsilons_vals
        
        c = keras.models.clone_model(last_layer_model)
        c.set_weights([new_weights])
        new_out = c.predict(lastlayer_inputs)
        print(network.outputVars)
        print(new_out)
        
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkVerification3(epsilon_max, epsilon_interval)
    problem.run(model_name)