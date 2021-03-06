#!/usr/bin/env python3

from functools import reduce

from keras import backend as K
from keras.layers import Conv2D, Conv3D, Flatten, Input, Reshape, Lambda, concatenate
from keras.layers import MaxPooling3D, UpSampling2D,UpSampling3D
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.utils.vis_utils as vis

import utils.pkl_xz as pkl_xz

apply = lambda f, x: f(x)
def flip(f):
    return lambda a, b: f(b, a)
apply_sequence = lambda l, x: reduce(flip(apply), l, x)

data_format = "channels_first"

BN = BatchNormalization
C3 = lambda filter_size: Conv3D(
        filter_size,
        (3, 3, 3),
        data_format=data_format,
        activation="relu",
        padding="same")
def P3(shape=(2, 2, 2),strides=None):
    return MaxPooling3D(
        shape,
        strides=strides,
        data_format=data_format)
C2 = lambda filter_size: Conv2D(
        filter_size,
        (3,3),
        data_format=data_format,
        padding="same")
U2 = lambda: UpSampling2D(data_format=data_format)

coarse_architecture = [
    # encoder                              #112, 16
    C3(64), P3((1,2,2)),                   #56 , 16
    C3(128), P3(),                         #28 , 8
    C3(256), C3(256), P3(),                #14 , 4
    C3(512), C3(512), P3(strides=(4,2,2)), #7  , 1
    # decoder
    Reshape((512,7,7)),
    BN(), C2(256), LeakyReLU(0.001), U2(), #14
    BN(), C2(128), LeakyReLU(0.001), U2(), #28
    BN(), C2(64),  LeakyReLU(0.001), U2(), #56
    BN(), C2(32),  LeakyReLU(0.001), U2(), #112
    BN(), C2(16),  LeakyReLU(0.001),
    BN(), C2(1), Activation("relu"),
    Reshape((112,112))
]

C2 = lambda f: Conv2D(f,(3,3),data_format=data_format,activation="relu",padding="same")
fine_architecture = [
    C2(32),
    C2(64), LeakyReLU(0.001),
    C2(32), LeakyReLU(0.001),
    C2(32), LeakyReLU(0.001),
    C2(16), LeakyReLU(0.001),
    C2(4), LeakyReLU(0.001),
    C2(1), Activation("relu"),
    Reshape((448,448),name="fine_output")
]

def coarse_inference(x):
    return apply_sequence(coarse_architecture, x)

def predict_model(weights_file):

    # Fine-tuned network output
    resized_input = Input(shape=(3,16,112,112),dtype='float32',name="resized_input")
    resized_output = coarse_inference(resized_input)

    last_frame = Input(shape=(3,448,448),dtype='float32',name="last_frame")
    resized_output = Reshape([1,]+resized_output.shape.as_list()[1:])(resized_output)
    resized_output = UpSampling2D(size=(4,4),data_format=data_format)(
            resized_output)

    fine_input = concatenate([last_frame,resized_output],axis=1)
    fine_output = apply_sequence(fine_architecture, fine_input)

    # Build model
    model = Model(inputs=[resized_input,last_frame],
                  outputs=[fine_output])

    model.load_weights(weights_file)

    return model

def model(weights_file = None):
    data_format = "channels_first"

    """
    def inference(examples):
        tensor = examples[0]
        """

    # Coarse network output
    cropped_input = Input(shape=(3,16,112,112),dtype='float32',name="cropped_input")
    cropped_output = coarse_inference(cropped_input)
    cropped_output = Activation("linear",name="coarse_output")(cropped_output)


    # Fine-tuned network output
    resized_input = Input(shape=(3,16,112,112),dtype='float32',name="resized_input")
    resized_output = coarse_inference(resized_input)

    last_frame = Input(shape=(3,448,448),dtype='float32',name="last_frame")
    resized_output = Reshape((1,112,112))(resized_output)
    resized_output = UpSampling2D(size=(4,4),data_format=data_format)(resized_output)

    fine_input = concatenate([last_frame,resized_output],axis=1)
    fine_output = apply_sequence(fine_architecture, fine_input)

    # Build model
    model = Model(inputs=[cropped_input,resized_input,last_frame],
                  outputs=[cropped_output,fine_output])

    if weights_file is None:
        c3d_params = pkl_xz.load("c3d_weights.pkl.xz")
        pretrained_layers = [2,4,6,7,9,10]
        for l, p in zip(pretrained_layers, c3d_params):
            model.layers[l].trainable=False
            model.layers[l].set_weights(p)
        del c3d_params
    else:
        model.load_weights(weights_file)

    return model
