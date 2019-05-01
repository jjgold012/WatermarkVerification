# from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils

sat = 'SAT'
unsat = 'UNSAT'

def findEpsilonInterval(epsilon_max, epsilon_interval, network, prediction):
    sat_epsilon = epsilon_max
    unsat_epsilon = 0.0
    sat_vals = None
    epsilon = sat_epsilon
    while abs(sat_epsilon - unsat_epsilon) > epsilon_interval:
        status, vals, out = evaluateEpsilon(epsilon, deepcopy(network), prediction)
        if status == sat:
            sat_epsilon = epsilon
            sat_vals = (status, vals, out)
        else:
            unsat_epsilon = epsilon
        epsilon = (sat_epsilon + unsat_epsilon)/2
    return unsat_epsilon, sat_epsilon , sat_vals


def evaluateEpsilon(epsilon, network, prediction):
    outputVars = network.outputVars[0]
    vals = dict()
    for out in range(len(outputVars)):
        if out != prediction:
            vals[out] = evaluateSingleOutput(epsilon, deepcopy(network), prediction, out)
            if vals[out][0]:
                return sat, vals, out
    return unsat, vals, -1
def evaluateSingleOutput(epsilon, network, prediction, output):
    # epsilon = epsilon_max
    outputVars = network.outputVars[0]
    for k in network.matMulLayers.keys():
        n, m = network.matMulLayers[k]['vals'].shape
        print(n,m)
        for i in range(n):
            for j in range(m):
                network.setUpperBound(network.matMulLayers[k]['vars'][i][j], network.matMulLayers[k]['vals'][i][j] + epsilon)
                network.setLowerBound(network.matMulLayers[k]['vars'][i][j], network.matMulLayers[k]['vals'][i][j] - epsilon)
        for i in range(len(network.biasAddLayers[k]['vals'])):
            network.setUpperBound(network.biasAddLayers[k]['vars'][i], network.biasAddLayers[k]['vals'][i] + epsilon)
            network.setLowerBound(network.biasAddLayers[k]['vars'][i], network.biasAddLayers[k]['vals'][i] - epsilon)
    MarabouUtils.addInequality(network, [outputVars[prediction], outputVars[output]], [1, -1], 0)
    return network.solve(verbose=False)


def run(args):
    model_name = args.model
    net_model, submodel, last_layer_model = utils.splitModel(model_name)
    filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
    
    inputs = np.load(args.input_path)
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    epsilon_vals = list()
    for i in range(len(inputs)):

        input_test = np.reshape(inputs[i], (1,28,28,1))
        
        prediction = np.argmax(net_model.predict(input_test))
        network = Marabou.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
        
        unsat_epsilon, sat_epsilon, sat_vals = findEpsilonInterval(epsilon_max, epsilon_interval, network, prediction)
        epsilon_vals.append((unsat_epsilon, sat_epsilon, prediction, sat_vals))
    
    epsilon_vals.sort(key=lambda t: t[0])
    out_file = open("out.txt", "w")
    for i in range(len(inputs)):
        out_file.write('epsilon: ({}, {}), original prediction: {}, match prediction: {}\n'.format(epsilon_vals[i][0], epsilon_vals[i][1], epsilon_vals[i][2], epsilon_vals[i][3][2]))
    out_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='../nn_verification/data/wm.set.npy', help='input file path')
    parser.add_argument('--epsilon_max', default=100, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    run(args)