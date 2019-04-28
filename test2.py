# from nn_verification.utils import load_model
import numpy as np
import os
import sys
sys.path.insert(0,'..')
import uuid
from nn_verification.utils import load_model
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from maraboupy import Marabou

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    args = parser.parse_args()

    MODELS_PATH = '../Models'
    model_name = args.model
    net_model = load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    print('Model Summary')
    print(net_model.summary())

    submodel = keras.Model(inputs=net_model.inputs, outputs=net_model.layers[-2].output, name='submodel-'+str(uuid.uuid4())[:5])
    print('SubModel Summary')
    print(submodel.summary())

    input_test = np.reshape(np.array(range(784))/(1000), (1,28,28,1))
    wm_images = np.load('../nn_verification/data/wm.set.npy')  
    input_test = np.reshape(wm_images[1], (1,28,28,1))
    print(submodel.predict(input_test))

    in_shape = (submodel.output.shape[1].value,)
    print(in_shape)
    last_layer_in = keras.layers.Input(shape=in_shape, name='ll_input-'+str(uuid.uuid4())[:5])
    last_layer_model = keras.Model(inputs=last_layer_in, outputs=net_model.layers[-1](last_layer_in), name='lastlayer-'+str(uuid.uuid4())[:5])
    print('Last Layer Summary')
    print(last_layer_model.summary())

    print(last_layer_model.predict(submodel.predict(input_test)))
    print(net_model.predict(input_test))

    sess = K.get_session()
    file_writer = tf.summary.FileWriter('./logs', sess.graph)
    output_node_names = [node.name[:-2] for node in last_layer_model.outputs]
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
    graph_io.write_graph(constant_graph, '../ProtobufNetworks', '{}.pb'.format(model_name), as_text=False)
    print('saved the constant graph (ready for inference) at: ', os.path.join('ProtobufNetworks', '{}.pb'.format(model_name)))


    filename = '../ProtobufNetworks/{}.pb'.format(model_name)
   
    network = Marabou.read_tf_weights_as_var(filename=filename, inputVals=submodel.predict(input_test))
    
    # network.setUpperBound(network.outputVars[0][0], 5)
    epsilon = 0
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