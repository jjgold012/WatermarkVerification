# from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import utils
from maraboupy import Marabou
from maraboupy import MarabouUtils

def run(args):
    model_name = args.model
    net_model, submodel, last_layer_model = utils.splitModel(model_name)
    
    wm_images = np.load(args.input_path)  
    input_test = np.reshape(wm_images[1], (1,28,28,1))
    
    # print(last_layer_model.predict(submodel.predict(input_test)))
    # print(net_model.predict(input_test))

    filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
    network = Marabou.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
    
    # network.setUpperBound(network.outputVars[0][0], 5)
    epsilon = float(args.epsilon)
    for k in network.matMulLayers.keys():
        n, m = network.matMulLayers[k]['vals'].shape
        print(n,m)
        for i in range(n):
            for j in range(m):

            # # for i3 in network.inputVars[0][0][i1][i2]:
                network.setUpperBound(network.matMulLayers[k]['vars'][i][j], network.matMulLayers[k]['vals'][i][j] + epsilon)
                network.setLowerBound(network.matMulLayers[k]['vars'][i][j], network.matMulLayers[k]['vals'][i][j] - epsilon)
        for i in range(len(network.biasAddLayers[k]['vals'])):
            network.setUpperBound(network.biasAddLayers[k]['vars'][i], network.biasAddLayers[k]['vals'][i] + epsilon)
            network.setLowerBound(network.biasAddLayers[k]['vars'][i], network.biasAddLayers[k]['vals'][i] - epsilon)
    vals, stats = network.solve()
    print(vals)
    print(stats)

    print('inputVars')
    print(network.inputVars)
    print('inputVals')
    print(network.inputVals)
    print('layersMatMul')
    print(network.matMulLayers)
    print('layersBiasAdd')
    print(network.biasAddLayers)
    print('numOfLayers')
    print(network.numOfLayers)
    print('numVars')
    print(network.numVars)

    print(len(network.equList))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='../nn_verification/data/wm.set.npy', help='input file path')
    parser.add_argument('--epsilon', default=0.0, help='epsilon to use')
    parser.add_argument('--epsilon_interval', default=0.01, help='epsilon smallest change')
    args = parser.parse_args()
    run(args)