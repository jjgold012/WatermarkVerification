from nn_verification.utils import load_model
import numpy as np
import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from maraboupy import Marabou

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=' Construct Siamese Network.')
    parser.add_argument('--proto', help='the name of the Protobuf file')
    parser.add_argument('--model', help='the name of the model')
    args = parser.parse_args()

    MODELS_PATH = './Models'
    model_name = args.model
    net_model = load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    sess = K.get_session()
    file_writer = tf.summary.FileWriter('./logs', sess.graph)
    output_node_names = [node.name[:-2] for node in net_model.outputs]
    print('output node names: ', output_node_names)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)
    graph_io.write_graph(constant_graph, 'ProtobufNetworks', '{}.pb'.format(model_name), as_text=False)
    print('saved the constant graph (ready for inference) at: ', os.path.join('ProtobufNetworks', '{}.pb'.format(args.proto)))


    filename = './ProtobufNetworks/{}.pb'.format(model_name)
    # with tf.gfile.GFile(filename, "rb") as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    # with tf.Graph().as_default() as graph:
    #     tf.import_graph_def(graph_def, name="")
    # sess = tf.Session(graph=graph)
    # file_writer = tf.summary.FileWriter('./logs', sess.graph)

    network = Marabou.read_tf(filename)
    
    # network.setUpperBound(network.outputVars[0][0], 5)
    for i1 in range(len(network.inputVars[0][0])):
        for i2 in network.inputVars[0][0][i1]:
            # for i3 in network.inputVars[0][0][i1][i2]:
            network.setUpperBound(i2, 1)
            network.setLowerBound(i2, 0)
    vals, stats = network.solve()

