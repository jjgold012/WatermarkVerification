import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from WatermarkVerification1 import *
import MarabouNetworkTFWeightsAsVar
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkVerification3(WatermarkVerification):

    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    def evaluateSingleOutput(self, epsilon, network, prediction, output):
        outputVars = network.outputVars[0]
        abs_epsilons = list()
        for k in network.matMulLayers.keys():
            n, m = network.matMulLayers[k]['vals'].shape
            print(n,m)
            for i in range(n):
                for j in range(m):
                    epsilon_var = network.matMulLayers[k]['epsilons'][i][j]
                    network.setUpperBound(epsilon_var, epsilon)
                    network.setLowerBound(epsilon_var, -epsilon)
                    abs_epsilon_var = self.epsilonABS(network, epsilon_var)
                    abs_epsilons.append(abs_epsilon_var)

        e = MarabouUtils.Equation(EquationType=MarabouCore.Equation.LE)
        for i in range(len(abs_epsilons)):
            e.addAddend(1, abs_epsilons[i])
        e.setScalar(epsilon)
        network.addEquation(e)

        MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)


        return network.solve(verbose=True)


    def run(self, model_name):
       
        submodel, last_layer_model = utils.splitModel(self.net_model)
        filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
        
        
        epsilon_vals = list()
        # num_of_inputs_to_run = len(self.inputs)
        num_of_inputs_to_run = 2
        input_test = np.reshape(self.inputs, (self.inputs.shape[0], self.inputs.shape[1], self.inputs.shape[2], 1))
            
        prediction = self.net_model.predict(input_test)
        network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
        
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
        epsilon_vals.append((unsat_epsilon, sat_epsilon, prediction, sat_vals))
        
        
        
        epsilon_vals.sort(key=lambda t: t[0])
        out_file = open("WatermarkVerification3.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        for i in range(num_of_inputs_to_run):
            out_file.write('{},{},{},{}\n'.format(epsilon_vals[i][0], epsilon_vals[i][1], epsilon_vals[i][2], epsilon_vals[i][3][2]))
        out_file.close()


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='../nn-verification/data/wm.set.npy', help='input file path')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    inputs = np.load(args.input_path)
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    model_name = args.model
    MODELS_PATH = './Models'
    net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    problem = WatermarkVerification3(net_model, epsilon_max, epsilon_interval, inputs)
    problem.run(model_name)