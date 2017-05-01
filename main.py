
from __future__ import print_function
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile,exception_verbosity=high"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import sys
import config
from coco_loader import coco_loader
from coco_training import train_coco
from config import TRAINED_MODEL_FILE_1, TRAINED_MODEL_FILE_2, TRAINED_MODEL_FILE_3, TRAINED_MODEL_FILE_4, OUTPUT_FOLDER_TRAIN_1, OUTPUT_FOLDER_TRAIN_2, OUTPUT_FOLDER_TRAIN_3, OUTPUT_FOLDER_TRAIN_4, OUTPUT_FOLDER_VALID_1, OUTPUT_FOLDER_VALID_2, OUTPUT_FOLDER_VALID_3, OUTPUT_FOLDER_VALID_4
from predict import predict

if __name__ == '__main__':

    #train_length = 82782
    #valid_length = 40504
    train_length = 9969
    valid_length = 9981

    if not os.path.isdir("trained_models"):
        os.makedirs("trained_models")

    

    #LEARN 1
    train_set_1 = coco_loader(input="train2014_9969samples", target="train2014_9969samples")
    valid_set_1 = coco_loader(input="val2014_9981samples", target="val2014_9981samples")
    train_coco(train_set_1, valid_set_1, network_type=1, depth_input=16, depth_predicted=20, trained_model_file=TRAINED_MODEL_FILE_1, stage=1)

    train_input_1, train_target, train_caption, train_cap_id = train_set_1.load_items(0, train_length, depth_input=16, transpose_y=False)
    valid_input_1, valid_target, valid_caption, valid_cap_id = valid_set_1.load_items(0, valid_length, depth_input=16, transpose_y=False)


    #create output folders
    if not os.path.isdir(OUTPUT_FOLDER_TRAIN_1):
        os.makedirs(OUTPUT_FOLDER_TRAIN_1)
    if not os.path.isdir(OUTPUT_FOLDER_VALID_1):
        os.makedirs(OUTPUT_FOLDER_VALID_1)

    #PREDICT 1
    predict(train_input_1, train_target, train_cap_id, len(train_input_1), depth_predicted=20, trained_model_file=TRAINED_MODEL_FILE_1, output_folder=OUTPUT_FOLDER_TRAIN_1)
    predict(valid_input_1, valid_target, valid_cap_id, len(valid_input_1), depth_predicted=20, trained_model_file=TRAINED_MODEL_FILE_1, output_folder=OUTPUT_FOLDER_VALID_1)
    #predict(train_input_1, train_target, train_cap_id, 10, depth_predicted=24, trained_model_file=TRAINED_MODEL_FILE_1, output_folder=OUTPUT_FOLDER_TRAIN_1)
    #predict(valid_input_1, valid_target, valid_cap_id, 10, depth_predicted=24, trained_model_file=TRAINED_MODEL_FILE_1, output_folder=OUTPUT_FOLDER_VALID_1)


    #for i in range(10):
    #    print(valid_caption[i])



    #LEARN 2
    train_set_2 = coco_loader(input=OUTPUT_FOLDER_TRAIN_1, target="train2014_9969samples")
    valid_set_2 = coco_loader(input=OUTPUT_FOLDER_VALID_1, target="val2014_9981samples")
    train_coco(train_set_2, valid_set_2, network_type=1, depth_input=20, depth_predicted=24, trained_model_file=TRAINED_MODEL_FILE_2, stage=2)

    train_input_2, train_target, train_caption, train_cap_id = train_set_2.load_items(0, train_length, depth_input=20, transpose_y=False)
    valid_input_2, valid_target, valid_caption, valid_cap_id = valid_set_2.load_items(0, valid_length, depth_input=20, transpose_y=False)

    #create output folder
    if not os.path.isdir(OUTPUT_FOLDER_TRAIN_2):
        os.makedirs(OUTPUT_FOLDER_TRAIN_2)
    if not os.path.isdir(OUTPUT_FOLDER_VALID_2):
        os.makedirs(OUTPUT_FOLDER_VALID_2)

    #PREDICT 2
    predict(train_input_2, train_target, train_cap_id, train_length, depth_predicted=24, trained_model_file=TRAINED_MODEL_FILE_2, output_folder=OUTPUT_FOLDER_TRAIN_2)
    predict(valid_input_2, valid_target, valid_cap_id, valid_length, depth_predicted=24, trained_model_file=TRAINED_MODEL_FILE_2, output_folder=OUTPUT_FOLDER_VALID_2)




    #LEARN 3
    train_set_3 = coco_loader(input=OUTPUT_FOLDER_TRAIN_2, target="train2014_9969samples")
    valid_set_3 = coco_loader(input=OUTPUT_FOLDER_VALID_2, target="val2014_9981samples")
    train_coco(train_set_3, valid_set_3, network_type=1, depth_input=24, depth_predicted=28, trained_model_file=TRAINED_MODEL_FILE_3, stage=3)

    train_input_3, train_target, train_caption, train_cap_id = train_set_3.load_items(0, train_length, depth_input=24, transpose_y=False)
    valid_input_3, valid_target, valid_caption, valid_cap_id = valid_set_3.load_items(0, valid_length, depth_input=24, transpose_y=False)

    #create output folder
    if not os.path.isdir(OUTPUT_FOLDER_TRAIN_3):
        os.makedirs(OUTPUT_FOLDER_TRAIN_3)
    if not os.path.isdir(OUTPUT_FOLDER_VALID_3):
        os.makedirs(OUTPUT_FOLDER_VALID_3)

    #PREDICT 3
    predict(train_input_3, train_target, train_cap_id, train_length, depth_predicted=28, trained_model_file=TRAINED_MODEL_FILE_3, output_folder=OUTPUT_FOLDER_TRAIN_3)
    predict(valid_input_3, valid_target, valid_cap_id, valid_length, depth_predicted=28, trained_model_file=TRAINED_MODEL_FILE_3, output_folder=OUTPUT_FOLDER_VALID_3)




    #LEARN 4
    train_set_4 = coco_loader(input=OUTPUT_FOLDER_TRAIN_3, target="train2014_9969samples")
    valid_set_4 = coco_loader(input=OUTPUT_FOLDER_VALID_3, target="val2014_9981samples")
    train_coco(train_set_4, valid_set_4, network_type=1, depth_input=28, depth_predicted=32, trained_model_file=TRAINED_MODEL_FILE_4, stage=4)

    train_input_4, train_target, train_caption, train_cap_id = train_set_4.load_items(0, train_length, depth_input=28, transpose_y=False)
    valid_input_4, valid_target, valid_caption, valid_cap_id = valid_set_4.load_items(0, valid_length, depth_input=28, transpose_y=False)

    #create output folder
    if not os.path.isdir(OUTPUT_FOLDER_TRAIN_4):
        os.makedirs(OUTPUT_FOLDER_TRAIN_4)
    if not os.path.isdir(OUTPUT_FOLDER_VALID_4):
        os.makedirs(OUTPUT_FOLDER_VALID_4)

    #PREDICT 4
    predict(train_input_4, train_target, train_cap_id, train_length, depth_predicted=32, trained_model_file=TRAINED_MODEL_FILE_4, output_folder=OUTPUT_FOLDER_TRAIN_4)
    predict(valid_input_4, valid_target, valid_cap_id, valid_length, depth_predicted=32, trained_model_file=TRAINED_MODEL_FILE_4, output_folder=OUTPUT_FOLDER_VALID_4)
