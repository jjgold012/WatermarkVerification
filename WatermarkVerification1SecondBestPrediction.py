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

    def __init__(self, epsilon_max, epsilon_interval):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval

    def findEpsilonInterval(self, network, prediction, secondBestPrediction):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals, out = self.evaluateEpsilon(epsilon, deepcopy(network), prediction, secondBestPrediction)
            if status == sat:
                sat_epsilon = epsilon
                sat_vals = (status, vals, out)
            else:
                unsat_epsilon = epsilon
            epsilon = (sat_epsilon + unsat_epsilon)/2
        return unsat_epsilon, sat_epsilon , sat_vals


    def evaluateEpsilon(self, epsilon, network, prediction, secondBestPrediction):        
        vals = self.evaluateSingleOutput(epsilon, deepcopy(network), prediction, secondBestPrediction)
        if vals[0]:
            return sat, vals, secondBestPrediction
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
       
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)

        out_file = open("./data/results/WatermarkVerification1SecondBestPrediction.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,second-best-prediction\n')
        out_file.flush()
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))
        predIndices = np.flip(np.argsort(predictions, axis=1), axis=1)        
        num_of_inputs_to_run = lastlayer_inputs.shape[0]
        # num_of_inputs_to_run = 20
        epsilons_vals = np.array([])
        for i in range(num_of_inputs_to_run):
            prediction = np.argmax(predictions[i])
            prediction = predIndices[i][0]
            secondBestPrediction = predIndices[i][1]
            inputVals = np.reshape(lastlayer_inputs[i], (1, lastlayer_inputs[i].shape[0]))
            network = MarabouNetworkTFWeightsAsVar.read_tf_weights_as_var(filename=filename, inputVals=inputVals)
            
            unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, prediction, secondBestPrediction)
            out_file.write('{},{},{},{}\n'.format(unsat_epsilon, sat_epsilon, prediction, sat_vals[2]))
            out_file.flush()

            all_vals = sat_vals[1][0]
            epsilons_vars = network.matMulLayers[0]['epsilons']
            newVars = np.array([[all_vals[epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
            newVars = np.reshape(newVars, (1, newVars.shape[0], newVars.shape[1]))
            epsilons_vals = newVars if epsilons_vals.size==0 else np.append(epsilons_vals, newVars, axis=0)
        
        out_file.close()
        np.save('./data/results/{}.WatermarkVerification1.vals'.format(model_name), epsilons_vals)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--epsilon_max', default=1, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    model_name = args.model
    MODELS_PATH = './Models'
    problem = WatermarkVerification(epsilon_max, epsilon_interval)
    problem.run(model_name)