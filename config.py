# -*- coding: utf-8 -*-
"""
Created on Sat Mar 04 22:12:27 2017

@author: OKToffa
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"


TRAINED_MODEL_FILE = 'trainedmodel.pkl'    # File to store the learned model
TRAINED_MODEL_FILE_1 = 'trainedmodel_1.pkl'    # File to store the learned model
TRAINED_MODEL_FILE_2 = 'trainedmodel_2.pkl'    # File to store the learned model
TRAINED_MODEL_FILE_3 = 'trainedmodel_3.pkl'    # File to store the learned model
TRAINED_MODEL_FILE_4 = 'trainedmodel_4.pkl'    # File to store the learned model

MODELS_FOLDER = 'trained_models'

OUTPUT_FOLDER_TRAIN_1 = 'predicted_train_1'
OUTPUT_FOLDER_TRAIN_2 = 'predicted_train_2'
OUTPUT_FOLDER_TRAIN_3 = 'predicted_train_3'
OUTPUT_FOLDER_TRAIN_4 = 'predicted_train_4'
OUTPUT_FOLDER_VALID_1 = 'predicted_valid_1'
OUTPUT_FOLDER_VALID_2 = 'predicted_valid_2'
OUTPUT_FOLDER_VALID_3 = 'predicted_valid_3'
OUTPUT_FOLDER_VALID_4 = 'predicted_valid_4'

TRAINING_EPOCHS = 10
BATCH_SIZE = 100

NUM_HIDDEN = 3*8*8
INPUT_SHAPE = (None,3,64,64)
