
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import pickle as pkl
import lasagne
import theano
import numpy as np
#import matplotlib.pyplot as plt
from config import TRAINED_MODEL_FILE, MODELS_FOLDER
import PIL.Image as Image

def predict(test_set_x, test_set_y, cap_id, test_size, depth_predicted, trained_model_file, output_folder):
    print('loading the trained model')
    #with open(TRAINED_MODEL_FILE, 'rb') as f:
        #u = pkl._Unpickler(f)
        #u.encoding = 'latin1'
        #classifier = u.load()
    #classifier = pkl.load(open(TRAINED_MODEL_FILE, 'rb'), encoding='latin1')

    os.chdir(MODELS_FOLDER)
    print('compiling the predictor function')
    model = pkl.load(open(trained_model_file))
    model_values = pkl.load(open("trainedmodel_1_values.pkl"))
    lasagne.layers.set_all_param_values(model.network, model_values)
    os.chdir('../')
    predict_model = theano.function(
        inputs=[model.input],
        outputs=lasagne.layers.get_output(model.network, deterministic=True))
    #test_set_x = test_set_x.get_value()
    predicted_values = predict_model(test_set_x[:test_size])

    print('printing the test values')
    #fig = plt.figure()
    for i in range(test_size):
        img_input = 255*test_set_y[i]
        img_input = img_input.astype('uint8')
        #plt.subplot(2, test_size, i+1)
        #plt.imshow(img_input)
        img_output_temp = 255*predicted_values[i].reshape(3,64,64)
        img_output_temp = np.transpose(img_output_temp, (1, 2, 0))
        img_output_temp = img_output_temp.astype('uint8')
        img_output = np.copy(img_input)
        #for u in range(32):
        #    for v in range(32):
        #        img_output[u+16,v+16,:] = img_output_temp[u+16,v+16,:]
        #        img_output[u+8,v+8,:] = 0
        img_output[16:48,16:48,:] = img_output_temp[16:48,16:48,:]
        if depth_predicted < 32:
            img_output[depth_predicted:64-depth_predicted,depth_predicted:64-depth_predicted,:] = 0
        #plt.subplot(2, test_size, test_size+i+1)
        #plt.imshow(img_output)
        os.chdir(output_folder)
        #Image.fromarray(img_input).save(cap_id[i]+".jpg")
        Image.fromarray(img_output).save(cap_id[i]+".jpg")
        os.chdir('../')
    #plt.show()
    #plt.savefig("result.png", dpi=fig.dpi)

    #return predicted_values
