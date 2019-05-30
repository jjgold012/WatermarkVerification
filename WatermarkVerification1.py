# from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
import MarabouNetworkTFWeightsAsVar
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkVerification:

    def __init__(self, net_model, epsilon_max, epsilon_interval, inputs):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval
        self.inputs = inputs
        self.net_model = net_model

    def findEpsilonInterval(self, network, prediction):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals, out = self.evaluateEpsilon(epsilon, deepcopy(network), prediction)
            if status == sat:
                sat_epsilon = epsilon
                sat_vals = (status, vals, out)
            else:
                unsat_epsilon = epsilon
            epsilon = (sat_epsilon + unsat_epsilon)/2
        return unsat_epsilon, sat_epsilon , sat_vals


    def evaluateEpsilon(self, epsilon, network, prediction):
        outputVars = network.outputVars[0]
        vals = dict()
        for out in range(len(outputVars)):
            if out != prediction:
                vals[out] = self.evaluateSingleOutput(epsilon, deepcopy(network), prediction, out)
                if vals[out][0]:
                    return sat, vals, out
        return unsat, vals, -1


    def evaluateSingleOutput(self, epsilon, network, prediction, output):
        outputVars = network.outputVars[0]
        for k in network.matMulLayers.keys():
            n, m = network.matMulLayers[k]['vals'].shape
            print(n,m)
            for i in range(n):
                for j in range(m):
                    network.setUpperBound(network.matMulLayers[k]['epsilons'][i][j], epsilon)
                    network.setLowerBound(network.matMulLayers[k]['epsilons'][i][j], -epsilon)
            
        MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)
        return network.solve(verbose=False)


    def run(self, model_name):
       
        submodel, last_layer_model = utils.splitModel(self.net_model)
        filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
        
        
        epsilon_vals = list()
        for i in range(len(self.inputs)):

            input_test = np.reshape(self.inputs[i], (1, self.inputs.shape[1], self.inputs.shape[2], 1))
            
            prediction = np.argmax(self.net_model.predict(input_test))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            epsilon_vals.append((unsat_epsilon, sat_epsilon, prediction, sat_vals))
        
        epsilon_vals.sort(key=lambda t: t[0])
        out_file = open("WatermarkVerification1.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        for i in range(len(inputs)):
            out_file.write('{},{},{},{}\n'.format(epsilon_vals[i][0], epsilon_vals[i][1], epsilon_vals[i][2], epsilon_vals[i][3][2]))
        out_file.close()


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='../nn-verification/data/wm.set.npy', help='input file path')
    parser.add_argument('--epsilon_max', default=1, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    inputs = np.load(args.input_path)
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    model_name = args.model
    MODELS_PATH = './Models'
    net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    problem = WatermarkVerification(net_model, epsilon_max, epsilon_interval, inputs)
    problem.run(model_name)