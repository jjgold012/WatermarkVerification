import numpy as np
import os
import argparse
import utils
from pprint import pprint
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from tensorflow import keras

from WatermarkVerification1 import *
import MarabouNetworkTFWeightsAsVar2
sat = 'SAT'
unsat = 'UNSAT'

class WatermarkVerification4(WatermarkVerification):

    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouUtils.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouUtils.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    # def evaluateSingleOutput(self, epsilon, network, prediction, output):
    #     outputVars = network.outputVars[0]
    #     abs_epsilons = list()
    #     for k in network.matMulLayers.keys():
    #         n, m = network.matMulLayers[k]['vals'].shape
    #         print(n,m)
    #         for i in range(n):
    #             for j in range(m):
    #                 epsilon_var = network.matMulLayers[k]['epsilons'][i][j]
    #                 network.setUpperBound(epsilon_var, epsilon)
    #                 network.setLowerBound(epsilon_var, -epsilon)
    #                 abs_epsilon_var = self.epsilonABS(network, epsilon_var)
    #                 abs_epsilons.append(abs_epsilon_var)

    #     e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
    #     for i in range(len(abs_epsilons)):
    #         e.addAddend(1, abs_epsilons[i])
    #     e.setScalar(epsilon)
    #     network.addEquation(e)

    #     MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)


    #     return network.solve(verbose=True)

    def evaluateEpsilon(self, epsilon, network, prediction):
        outputVars = network.outputVars
        abs_epsilons = list()
        n, m = network.epsilons.shape
        print(n,m)
        for i in range(n):
            for j in range(m):
                epsilon_var = network.epsilons[i][j]
                network.setUpperBound(epsilon_var, epsilon)
                network.setLowerBound(epsilon_var, -epsilon)
                abs_epsilon_var = self.epsilonABS(network, epsilon_var)
                abs_epsilons.append(abs_epsilon_var)

        e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
        for i in range(len(abs_epsilons)):
            e.addAddend(1, abs_epsilons[i])
        e.setScalar(epsilon)
        network.addEquation(e)

        predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)        
        for i in range(outputVars.shape[0]):
            maxPred = predIndices[i][0]
            secondMaxPred = predIndices[i][1]
            MarabouUtils.addInequality(network, [outputVars[i][maxPred], outputVars[i][secondMaxPred]], [1, -1], 0)
        
        options = Marabou.createOptions(dnc=True)
        stats = network.solve(verbose=False, options=options)
        newOut = predIndices[:,1]
        if stats[0]:
            return sat, stats, newOut
        else:
            return unsat, stats, newOut

    def run(self, model_name, numOfInputs):        
        
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))[:numOfInputs]
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))[:numOfInputs]
        network = MarabouNetworkTFWeightsAsVar2.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, predictions)
        epsilons_vars = network.epsilons
        epsilons_vals = np.array([[sat_vals[1][0][epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
        # newVars = np.reshape(newVars, (1, newVars.shape[0], newVars.shape[1]))
        maxPred = np.argmax(predictions, axis=1)

        out_file = open('./data/results/problem4/{}.WatermarkVerification4.{}.wm.out'.format(model_name, numOfInputs), 'w')
        out_file.write('unsat_epsilon: {}\n'.format(unsat_epsilon))
        out_file.write('sat_epsilon: {}\n'.format(sat_epsilon))
        out_file.write('\noriginal prediction: \n')
        pprint(predictions.tolist(), out_file)
        out_file.write('\nmax prediction: \n')
        pprint(maxPred.tolist(), out_file)
        out_file.write('\nnew prediction: \n')
        pprint(sat_vals[2].tolist(), out_file)
        
        np.save('./data/results/problem4/{}.WatermarkVerification4.{}.wm.vals'.format(model_name, numOfInputs), epsilons_vals)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=5, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    parser.add_argument('--num_of_inputs', default=2, help='the number of inputs that needs to be falsify')
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    numOfInputs = int(args.num_of_inputs)
    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkVerification4(epsilon_max, epsilon_interval)
    problem.run(model_name, numOfInputs)