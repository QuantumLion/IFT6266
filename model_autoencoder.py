# -*- coding: utf-8 -*-
"""
Created on Thu Mar 02 22:58:41 2017

@author: OKToffa
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import lasagne
import theano.tensor as T
import numpy as np
import scipy.sparse
from functools import reduce
from config import BATCH_SIZE

class conv_autoencoder_seq(object):
    """Convolution Auto-Encoder classl"""

    def __init__(
        self,
        input=None,
        target=None,
        depth_input=16,
        depth_predicted=24,
        input_shape=(None,3,64,64),
        n_hidden=1000,
        n_convlayer=8,
        n_filter = 10,
        filter_sizex = 3,
        pool=2,
        #stride=1,
        nonlinearity = lasagne.nonlinearities.rectify
    ):
         # Parameters:
         #    input: Theano symbolic variable
         #    target: Theano symbolic variable
         #    input_shape : tuple of int or None elementsshape: tuple of int
         #    n_convlayer : number of convolution and pooling layer.
         #    n_hidden : int The number of units of the hidden layer
         #    nonlinearity : callable or None. applied to the activation layer.

        print("Init autoencoder network ...")
        self.input = input
        self.target = target
        self.depth_input = depth_input
        self.depth_predicted = depth_predicted


        #creating the encoder
        encoderLayer = lasagne.layers.InputLayer(shape=input_shape, input_var=input)
        # put some convolution layers with maxpooling
        image_size = input_shape[2]
        for i in range(n_convlayer):
            encoderLayer = lasagne.layers.Conv2DLayer(incoming = encoderLayer, num_filters=n_filter, filter_size=(filter_sizex, filter_sizex), pad='same', nonlinearity=nonlinearity)
            if (i == (n_convlayer/2)-1 or i == n_convlayer-1):
                encoderLayer = lasagne.layers.MaxPool2DLayer(incoming = encoderLayer, pool_size=(pool, pool))
                image_size = image_size/pool
                if i == 1:
                    n_filter = n_filter*4

        image_size = int(image_size)
        # put a fully connected layer
        encoder_input_shape = (-1, n_filter, image_size, image_size) #lasagne.layers.get_output_shape(convlayer)
        n_decoder_units = n_filter*image_size*image_size
        encoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_hidden, nonlinearity=nonlinearity)

        #creating the decoder
        decoderLayer = lasagne.layers.DenseLayer(incoming = encoderLayer, num_units = n_decoder_units, W = encoderLayer.W.T, nonlinearity=nonlinearity)

        #since the result of the decoderlayer is flatten, we have to unflatten it
        decoderLayer = lasagne.layers.ReshapeLayer(incoming = decoderLayer, shape = encoder_input_shape)
        # do the inverse operation of the convolution layers with maxunpooling
        for j in range(n_convlayer-1):
            if (j == 0 or j == n_convlayer/2-1):
                if j == n_convlayer/2-1:
                    n_filter = n_filter/4
                image_size = image_size/pool
                decoderLayer = lasagne.layers.Upscale2DLayer(incoming = decoderLayer, scale_factor=pool)
            decoderLayer = lasagne.layers.TransposedConv2DLayer(incoming = decoderLayer, num_filters=n_filter, filter_size=(filter_sizex, filter_sizex), crop='same', nonlinearity=nonlinearity)

        #decoderLayer = lasagne.layers.Upscale2DLayer(incoming = decoderLayer, scale_factor=pool)
        decoderLayer = lasagne.layers.TransposedConv2DLayer(incoming = decoderLayer, num_filters=3, filter_size=(filter_sizex, filter_sizex), crop='same', nonlinearity=nonlinearity)

        self.network = decoderLayer
        self.image_size=image_size



    def get_cost_updates(self):
        """ This function computes the cost and the updates for one training step """

        weight_decay = 1e-5

        network_output = lasagne.layers.get_output(self.network)

        """
        cost = lasagne.objectives.squared_error(network_output, self.target).mean()
        #cost_large = lasagne.objectives.squared_error(network_output[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input], self.target[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input]).sum()

        """
        #"""
        cost = lasagne.objectives.squared_error(network_output, self.target)
        #cost_large = (lasagne.objectives.squared_error(network_output[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input], self.target[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input])).sum()
        cost_large = (lasagne.objectives.squared_error(network_output, self.target)).sum()

        if self.depth_predicted < 32:
            cost_small = (lasagne.objectives.squared_error(network_output[:,:,self.depth_predicted:64-self.depth_predicted,self.depth_predicted:64-self.depth_predicted], self.target[:,:,self.depth_predicted:64-self.depth_predicted,self.depth_predicted:64-self.depth_predicted]).sum())
            cost_size = BATCH_SIZE*3*(64**2 - (64-2*self.depth_predicted)**2)
            cost = (cost_large - cost_small) / cost_size
            #cost = (cost_large - cost_small)
        else:
            cost_size = BATCH_SIZE*3*(64**2 - (64-2*self.depth_predicted)**2)
            cost = (cost_large) / cost_size
            #cost = (cost_large)
        #"""

        #weights_l1 = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l1)
        #cost += weight_decay * weights_l1
        weights_l2 = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        cost += weight_decay * weights_l2

        all_params = lasagne.layers.get_all_params(self.network,trainable=True)

        updates = lasagne.updates.adam(cost, all_params) #lasagne.updates.adadelta(cost, all_params)
        #updates = lasagne.updates.rmsprop(cost, all_params) #lasagne.updates.adadelta(cost, all_params)
        #updates = lasagne.updates.sgd(cost, all_params, 0.01) #lasagne.updates.adadelta(cost, all_params)

        return (cost, updates)



    def get_validation_error(self):
        """ This function computes the cost and the updates for one training step """

        weight_decay = 1e-5

        network_output = lasagne.layers.get_output(self.network)

        """
        cost = lasagne.objectives.squared_error(network_output, self.target).mean()
        #cost_large = lasagne.objectives.squared_error(network_output[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input], self.target[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input]).sum()

        """
        #"""
        cost = lasagne.objectives.squared_error(network_output, self.target)
        #cost_large = (lasagne.objectives.squared_error(network_output[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input], self.target[:,:,self.depth_input:64-self.depth_input,self.depth_input:64-self.depth_input])).sum()
        cost_large = (lasagne.objectives.squared_error(network_output, self.target)).sum()

        if self.depth_predicted < 32:
            cost_small = (lasagne.objectives.squared_error(network_output[:,:,self.depth_predicted:64-self.depth_predicted,self.depth_predicted:64-self.depth_predicted], self.target[:,:,self.depth_predicted:64-self.depth_predicted,self.depth_predicted:64-self.depth_predicted]).sum())
            cost_size = (BATCH_SIZE)*3*(64**2 - (64-2*self.depth_predicted)**2)
            cost = (cost_large - cost_small) / cost_size
            #cost = (cost_large - cost_small)
        else:
            cost_size = (BATCH_SIZE)*3*(64**2 - (64-2*self.depth_predicted)**2)
            cost = (cost_large) / cost_size
            #cost = (cost_large)
        #"""

        #weights_l1 = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l1)
        #cost += weight_decay * weights_l1
        weights_l2 = lasagne.regularization.regularize_network_params(self.network, lasagne.regularization.l2)
        cost += weight_decay * weights_l2

        #all_params = lasagne.layers.get_all_params(self.network,trainable=True)
        #updates = lasagne.updates.adam(cost, all_params) #lasagne.updates.adadelta(cost, all_params)
        return (cost)
