import numpy as np
import os
import argparse
import utils
from copy import deepcopy
from maraboupy import Marabou
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
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