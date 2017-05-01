# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 13:50:55 2017

@author: OKToffa
"""
from __future__ import print_function

import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import timeit
import numpy
import lasagne
import theano
import theano.tensor as T
import pickle as pkl
from config import TRAINING_EPOCHS, BATCH_SIZE, MODELS_FOLDER, NUM_HIDDEN, TRAINED_MODEL_FILE, INPUT_SHAPE
from model_autoencoder import conv_autoencoder_seq

def train_coco(
    train_set,
    valid_set,
    network_type=1,
    depth_input=16,
    depth_predicted=16,
    train_epochs= TRAINING_EPOCHS,
    batch_size=BATCH_SIZE,
    output_folder=MODELS_FOLDER,
    trained_model_file = TRAINED_MODEL_FILE,
    stage=1
):

    # compute number of minibatches for training, validation and testing
    n_train_batches = len(train_set.input_imgs) // batch_size
    n_valid_batches = len(valid_set.input_imgs) // batch_size

    print("Building the model ...")

    # allocate symbolic variables for the data
    index = T.lscalar()

    # generate symbolic variables for input (input_miniBatch and target_miniBatch represent a
    # minibatch)
    input_miniBatch = T.tensor4('input_miniBatch')
    target_miniBatch = T.tensor4('target_miniBatch')

    if stage > 1:
        os.chdir(output_folder)
        model = pkl.load(open("trainedmodel_"+str(stage-1)+".pkl"))
        model_values = pkl.load(open("trainedmodel_"+str(stage-1)+"_values.pkl"))
        os.chdir('../')
        lasagne.layers.set_all_param_values(model.network, model_values)
        cost, updates = model.get_cost_updates()
        validation_error = model.get_validation_error()
        train_model = theano.function(
            inputs = [model.input, model.target],
            outputs = [cost],
            updates = updates,
           # givens={ x: inputvar, y : targetvar},
            on_unused_input = 'ignore',
            mode = 'FAST_RUN'
        )
        validate_model = theano.function(
            inputs = [model.input, model.target],
            outputs = [validation_error],
            on_unused_input = 'ignore',
            mode = 'FAST_RUN'
        )
    else:
        # construct the conv_autoencoder_seq class
        # Each image has size 64*64
        model = conv_autoencoder_seq(
            input=input_miniBatch,
            target=target_miniBatch,
            depth_input=depth_input,
            depth_predicted=depth_predicted,
            input_shape=INPUT_SHAPE,
            n_hidden=NUM_HIDDEN
        )
        cost, updates = model.get_cost_updates()
        validation_error = model.get_validation_error()
        train_model = theano.function(
            inputs = [input_miniBatch,target_miniBatch],
            outputs = [cost],
            updates = updates,
           # givens={ x: inputvar, y : targetvar},
            on_unused_input = 'ignore',
            mode = 'FAST_RUN'
        )
        validate_model = theano.function(
            inputs = [input_miniBatch,target_miniBatch],
            outputs = [validation_error],
            on_unused_input = 'ignore',
            mode = 'FAST_RUN'
        )


    # ***************************
    # early-stopping parameters
    # ***************************
    # look as this many examples regardless
    patience_max = 3
    # wait this much longer when a new best is found
    patience_current = 0
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995

    best_validation_cost = numpy.inf

    #create output folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    start_time = timeit.default_timer()

    print("Training the model ...")
    print('Epochs ', train_epochs)
    print('Batches number ', n_train_batches)
    print('Batches size ', batch_size)


    if stage > 1:
        os.chdir(output_folder)
        trained_model_values = pkl.load(open("trainedmodel_"+str(stage-1)+"_values.pkl"))
        os.chdir('../')
        lasagne.layers.set_all_param_values(model.network, trained_model_values)
        """
        cost_2, updates_2 = trained_model.get_cost_updates()
        train_model_2up = theano.function(
            inputs = [trained_model.input,trained_model.target],
            outputs = [cost_2],
            updates = updates_2,
           # givens={ x: inputvar, y : targetvar},
            on_unused_input = 'ignore',
            mode = 'FAST_RUN'
        )
        validate_model_2up = theano.function(
        inputs = [trained_model.input],
        outputs = lasagne.layers.get_output(trained_model.network, deterministic=True),
        on_unused_input = 'ignore',
        mode = 'FAST_RUN'
        )
        """



    done_looping = False
    epoch = 0

    while (epoch < train_epochs) and (not done_looping):

        epoch = epoch + 1

        # go through training set
        train_miniBatch_avg_cost = []
        valid_miniBatch_avg_cost = []
        for train_miniBatch_index in range(n_train_batches):
            train_set_input, train_set_target, train_set_caption, train_set_cap_id = train_set.load_items(train_miniBatch_index, batch_size, depth_input)
            this_train_miniBatch_avg_cost = train_model(train_set_input, train_set_target)
            train_miniBatch_avg_cost.append(this_train_miniBatch_avg_cost[0])
            print("trained miniBatch %d, cost "% train_miniBatch_index, numpy.mean(this_train_miniBatch_avg_cost, dtype='float64'))
        print('Training epoch %d, training cost ' % epoch, numpy.mean(train_miniBatch_avg_cost, dtype='float64'))

        # compute loss on validation set
        for valid_miniBatch_index in range(n_valid_batches):
            valid_set_input, valid_set_target, valid_set_caption, valid_set_cap_id = valid_set.load_items(valid_miniBatch_index, batch_size, depth_input)
            valid_miniBatch_cost = validate_model(valid_set_input, valid_set_target)
            valid_miniBatch_avg_cost.append(valid_miniBatch_cost)
        this_validation_cost = numpy.mean(valid_miniBatch_avg_cost, dtype='float64')
        print(
            'epoch %i, validation error %f' %
            ( epoch, this_validation_cost)
        )

        # if we got the best validation score until now
        if this_validation_cost < best_validation_cost:
            best_validation_cost = this_validation_cost
            patience_current = 0

            os.chdir(output_folder)
            with open(trained_model_file, 'wb') as f:
                pkl.dump(model, f)
            with open("trainedmodel_"+str(stage)+"_values.pkl", 'wb') as g:
                model_values = lasagne.layers.get_all_param_values(model.network)
                pkl.dump(model_values, g)
            os.chdir('../')
        else:
            patience_current = patience_current + 1

        # iteration number
        if patience_max <= patience_current:
            done_looping = True
            break

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)

    print('The training ran for %.2fm' % ((training_time) / 60.))
