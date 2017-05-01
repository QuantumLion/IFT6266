# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 00:45:16 2017

@author: OKToffa
"""
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import glob
import pickle as pkl
import numpy as np
import theano
import PIL.Image as Image
class coco_loader(object):

    def __init__(
        self,
        mscoco="C:/Users/user/project_dumesnil/",
        input="train2014",
        target="train2014",
        caption_path="C:/Users/user/project_dumesnil/dict_key_imgID_value_caps_train_and_valid.pkl"
    ):
         # Parameters:
         #    mscoco: string coco folder
         #    split : string training folder
         #    caption_path: string caption path

        print('Loading ' + input + ' data...')
        self.mscoco = mscoco
        self.input_path = os.path.join(mscoco, input)
        self.input_imgs = glob.glob(self.input_path + "/*.jpg")
        self.target_path = os.path.join(mscoco, target)
        self.target_imgs = glob.glob(self.target_path + "/*.jpg")
        caption_path = os.path.join(mscoco, caption_path)
        with open(caption_path, 'rb') as fd:
            caption_dict = pkl.load(fd)
        self.caption_dict = caption_dict
        self.x = np.array(0)
        self.y = np.array(0)



    def load_items(self, batch_idx, batch_size, depth_input, transpose_x=True, transpose_y=True):
        batch_input_imgs = self.input_imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_target_imgs = self.target_imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]
        res_input = [self.load_item(i, input_path, depth_input) for i, input_path in enumerate(batch_input_imgs)]
        res_target = [self.load_item(i, target_path, 32) for i, target_path in enumerate(batch_target_imgs)]
        #remove None and unzip the list
        self.x, cap_x, cap_id_x = zip(*[x for x in res_input if x is not None])
        self.y, cap_y, cap_id_y = zip(*[y for y in res_target if y is not None])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        if(transpose_x):
            self.x = self.x.transpose((0, 3, 1, 2))
        if(transpose_y):
            self.y = self.y.transpose((0, 3, 1, 2))
        #return theano.shared(np.array(self.x), borrow = True), theano.shared(np.array(self.y), borrow = True), cap

        return np.array(self.x), np.array(self.y), cap_x, cap_id_x



    def load_item(self, index, img_path, depth_input):
        img = Image.open(img_path)
        img_array = np.array(img)
        cap_id = os.path.basename(img_path)[:-4]

        # create 32x32 black squre in the middle of the image
        center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
        if len(img_array.shape) == 3:
            image = np.copy(img_array)
            if depth_input < 32:
                image[depth_input:64-depth_input, depth_input:64-depth_input, :] = 0
        else:
            # skip gray images
            return None
            #return the normalized values
        return image.astype('float32')/255., self.caption_dict[cap_id], cap_id
