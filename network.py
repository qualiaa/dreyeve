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


def model(weights_file = None):
    data_format = "channels_first"

    """
    def inference(examples):
        tensor = examples[0]
        """

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
        # encoder                        #112, 16
        C3(64), P3((1,2,2)),               #56 , 8
        C3(128), P3(),                   #28 , 4
        C3(256), C3(256), P3(),          #14 , 2
        C3(512), C3(512), P3(strides=(4,2,2)),          #7  , 1
        # decoder
        Reshape((512,7,7)),
        BN(), C2(256), LeakyReLU(0.001), U2(), #14
        BN(), C2(128), LeakyReLU(0.001), U2(), #28
        BN(), C2(64),  LeakyReLU(0.001), U2(), #56
        BN(), C2(32),  LeakyReLU(0.001), U2(), #112
        BN(), C2(16),  LeakyReLU(0.001),
        BN(), C2(1), LeakyReLU(0.001)
    ]

    C2 = lambda f: Conv2D(f,(3,3),data_format=data_format,activation="relu",padding="same")
    fine_architecture = [
        C2(32),
        C2(64), LeakyReLU(0.001),
        C2(32), LeakyReLU(0.001),
        C2(32), LeakyReLU(0.001),
        C2(16), LeakyReLU(0.001),
        C2(4), LeakyReLU(0.001),
        C2(1), Activation("relu")
    ]

    def coarse_inference(x):
        return apply_sequence(coarse_architecture, x)

    # Coarse network output
    cropped_input = Input(shape=(3,16,112,112),dtype='float32',name="cropped_input")
    cropped_output = coarse_inference(cropped_input)


    # Fine-tuned network output
    resized_input = Input(shape=(3,16,112,112),dtype='float32',name="resized_input")
    resized_output = coarse_inference(resized_input)
    take_last_frame = Lambda(lambda x: x[:,:,-1,:,:],output_shape = (3,448,448))

    full_input = Input(shape=(3,16,448,448),dtype='float32',name="full_input")
    last_frame = take_last_frame(full_input)
    resized_output = UpSampling2D(size=(4,4),data_format=data_format)(
            resized_output)

    fine_input = concatenate([last_frame,resized_output],axis=1)
    fine_output = apply_sequence(fine_architecture, fine_input)

    # Build model
    model = Model(inputs=[full_input,cropped_input,resized_input],
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
