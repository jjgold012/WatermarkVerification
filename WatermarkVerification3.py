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


    def evaluateEpsilon(self, epsilon, network, prediction):
        outputVars = network.outputVars
        vals = dict()
        n, m = network.epsilons.shape
        print(n,m)
        for i in range(n):
            for j in range(m):
                network.setUpperBound(network.epsilons[i][j], epsilon)
                network.setLowerBound(network.epsilons[i][j], -epsilon)

        predIndices = np.flip(np.argsort(prediction, axis=1), axis=1)        
        for i in range(outputVars.shape[0]):
            maxPred = predIndices[i][0]
            secondMaxPred = predIndices[i][1]
            MarabouUtils.addInequality(network, [outputVars[i][maxPred], outputVars[i][secondMaxPred]], [1, -1], 0)
        stats = network.solve(verbose=False)
        newOut = predIndices[:,1]
        if stats[0]:
            return sat, stats, newOut
        # for out in range(len(outputVars)):
        #     if out != prediction:
        #         vals[out] = self.evaluateSingleOutput(epsilon, deepcopy(network), prediction, out)
        #         if vals[out][0]:
        #             return sat, vals, out
        # return unsat, vals, -1


    def evaluateSingleOutput(self, epsilon, network, prediction, output):
        outputVars = network.outputVars
        for k in network.matMulLayers.keys():
            n, m = network.matMulLayers[k]['vals'].shape
            print(n,m)
            for i in range(n):
                for j in range(m):
                    network.setUpperBound(network.matMulLayers[k]['epsilons'][i][j], epsilon)
                    network.setLowerBound(network.matMulLayers[k]['epsilons'][i][j], -epsilon)
            
        MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)
        return network.solve(verbose=False)


    def run(self, model_name, numOfInputs):        
        
        filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
        
        out_file = open("WatermarkVerification3.csv", "w")
        out_file.write('unsat-epsilon,sat-epsilon,original-prediction,sat-prediction\n')
        out_file.flush()
        
        lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(model_name))[:numOfInputs]
        predictions = np.load('./data/{}.prediction.npy'.format(model_name))[:numOfInputs]
        network = MarabouNetworkTFWeightsAsVar2.read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)
        maxPred = np.max(predictions, axis=1)
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilonInterval(network, predictions)
        print(maxPred)
        print('----------------------------------------------------------')
        print('unsat_epsilon: ', unsat_epsilon)
        print('sat_epsilon: ', sat_epsilon)
        print(sat_vals[2])
        # for i in range(network.epsilons.shape[0]):
        #     for j in range(network.epsilons.shape[1]):
        #         network.setUpperBound(network.epsilons[i][j], 1)
        #         network.setLowerBound(network.epsilons[i][j], 1)
        # vals = network.solve()
        # print(predictions)

        # epsilons_vars = network.epsilons
            
        # epsilons_vals = np.array([[vals[0][epsilons_vars[j][i]] for i in range(epsilons_vars.shape[1])] for j in range(epsilons_vars.shape[0])])
        
        # net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
        # submodel, last_layer_model = utils.splitModel(net_model)
        # weights = last_layer_model.get_weights()[0]
        # new_weights = weights + epsilons_vals
        
        # c = keras.models.clone_model(last_layer_model)
        # c.set_weights([new_weights])
        # new_out = c.predict(lastlayer_inputs)
        # print(network.outputVars)
        # print(new_out)
        
    

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
    problem = WatermarkVerification3(epsilon_max, epsilon_interval)
    problem.run(model_name, numOfInputs)