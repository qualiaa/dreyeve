#!/usr/bin/env python3

from keras import backend as K
from keras.layers import Conv2D, Conv3D, Flatten, Input, Reshape, Lambda, concatenate
from keras.layers import MaxPooling3D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.utils.vis_utils as vis

from functools import reduce

apply = lambda f, x: f(x)
def flip(f):
    return lambda a, b: f(b, a)
apply_sequence = lambda l, x: reduce(flip(apply), l, x)

data_format = "channels_last"

C3 = lambda f: Conv3D(f,(3,3,3),data_format=data_format,activation="relu",padding="same")
P3 = lambda: MaxPooling3D(data_format=data_format)
C2 = lambda f: Conv2D(f,(3,3),data_format=data_format,padding="same")
U2 = lambda: UpSampling2D(data_format=data_format)

coarse_architecture = [
    # encoder
    C3(64), P3(),
    C3(128), P3(),
    C3(256), C3(256), P3(),
    C3(512), C3(512), P3(),
    # decoder
    Reshape((512,7,7)),
    C2(256), LeakyReLU(0.001), U2(),
    C2(128), LeakyReLU(0.001), U2(),
    C2(64),  LeakyReLU(0.001), U2(),
    C2(32),  LeakyReLU(0.001), U2(),
    C2(16),  LeakyReLU(0.001),
    # missing U2 here?
    C2(1), LeakyReLU(0.001)
]

C2 = lambda f: Conv2D(f,(3,3),data_format=data_format,activation="relu",padding="same")
fine_architecture = [
    C2(32),
    C2(64),
    C2(32),
    C2(32),
    C2(16),
    C2(4)
]

def coarse_inference(x):
    return apply_sequence(coarse_architecture, x)

# Siamese subnetwork
cropped_input = Input(shape=(3,16,112,112),dtype='float32',name="cropped_input")
resized_input = Input(shape=(3,16,112,112),dtype='float32',name="resized_input")

cropped_output = coarse_inference(cropped_input)
resized_output = coarse_inference(resized_input)

# Fine-tuning subnetwork
take_last_frame = Lambda(lambda x: x[:,:,-1,:,:],output_shape = (3,112,112))

last_frame = take_last_frame(resized_input)

fine_input = concatenate([resized_output,last_frame],axis=1)
fine_output = apply_sequence(fine_architecture, fine_input)

# Build model
model = Model(inputs=[cropped_input,resized_input],
              outputs=[resized_output,fine_output])

vis.plot_model(model,to_file="keras.png")
model.summary()
