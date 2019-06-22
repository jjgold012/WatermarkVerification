import numpy as np      
import utils    
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='the name of the model')
    parser.add_argument('--input_path', default='../nn-verification/data/wm.set.npy', help='input file path')
    args = parser.parse_args()
    inputs = np.load(args.input_path)
    model_name = args.model
    MODELS_PATH = './Models'
    net_model = utils.load_model(os.path.join(MODELS_PATH, model_name+'_model.json'), os.path.join(MODELS_PATH, model_name+'_model.h5'))
    
    submodel, last_layer_model = utils.splitModel(net_model)
    filename = utils.saveModelAsProtobuf(last_layer_model, 'last.layer.{}'.format(model_name))
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2], 1))

    prediction = net_model.predict(inputs)
    lastlayer_input = submodel.predict(inputs)

    np.save('./data/{}.prediction'.format(model_name), prediction)    
    np.save('./data/{}.lastlayer.input'.format(model_name), lastlayer_input)    
