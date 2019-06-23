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

class WatermarkVerification2(WatermarkVerification):

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
        print('Start the run\nmodel: {} \nepsilon_max {} \nepsilon_interval: {}'.format(model_name, self.epsilon_max, self.epsilon_interval))
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        out_file = open("WatermarkVerification2.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        out_file.flush()
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        # num_of_inputs_to_run = lastlayer_inputs.shape[0]
        num_of_inputs_to_run = 5
        for i in range(num_of_inputs_to_run):
            
            prediction = np.argmax(predictions[i])
            inputVals = np.reshape(lastlayer_inputs[i], (1, lastlayer_inputs[i].shape[0]))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=inputVals)
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction)
            out_file.write('{},{},{},{}\n'.format(unsat_epsilon, sat_epsilon, prediction, sat_vals[2]))
            out_file.flush()
        
        out_file.close()


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    print(args)
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkVerification2(epsilon_max, epsilon_interval)
    problem.run(model_name)