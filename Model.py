from keras.models import Model
from keras.layers import Input, Dropout, Dense, Conv2D, Flatten, Merge, Activation, Conv2DTranspose, concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import numpy as np
import cv2
from sklearn import preprocessing
import tensorflow as tf
from flowlib import *
# def flowNetSD(self):
#     cnnLayers = Sequential()
#     cnnLayers.add(Conv2D(64, (3, 3), name = 'conv0', strides = 1, padding='same', activation=LeakyReLU(), input_shape=(320, 1152, 6)))
#     cnnLayers.add(Conv2D(64, (3, 3), name = 'conv1', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(128, (3, 3), name = 'conv1_1', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(128, (3, 3), name = 'conv2', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(128, (3, 3), name = 'conv2_1', strides = 1, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(256, (3, 3), name = 'conv3', strides = 1, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(256, (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(512, (3, 3), name = 'conv4', strides = 1, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(512, (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(512, (3, 3), name = 'conv5', strides = 1, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(512, (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU()))

#     cnnLayers.add(Conv2D(1024, (3, 3), name = 'conv6', strides = 2, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Conv2D(1024, (3, 3), name = 'conv6_1', strides = 1, padding='same', activation=LeakyReLU()))
#     cnnLayers.add(Flatten())
#     cnnLayers.add(Dense(128, kernel_initializer="uniform", activation=LeakyReLU()))
#     cnnLayers.add(Dense(12, kernel_initializer = 'uniform'))
#     cnnLayers.compile(optimizer = 'sgd', loss = 'mae')
#     return cnnLayers

def setFNetWeight(model):
    #input1, input2, labels = getBatch(0, 1, 2001)
    fnsd = np.load('FlowNetSD_weights.npy')
    for i in range(0,13):
        model.layers[i+1].set_weights([fnsd[2*i],fnsd[2*i+1]])
    print 'conv layers finished'

    model.layers[14].set_weights([fnsd[27], fnsd[28]])
    model.layers[15].set_weights([fnsd[26]])
    model.layers[16].set_weights([fnsd[29]])

    model.layers[18].set_weights([fnsd[30], fnsd[31]])
    model.layers[19].set_weights([fnsd[32], fnsd[33]])
    model.layers[20].set_weights([fnsd[34]])
    model.layers[21].set_weights([fnsd[35]])

    model.layers[23].set_weights([fnsd[36], fnsd[37]])
    model.layers[24].set_weights([fnsd[38], fnsd[39]])
    model.layers[25].set_weights([fnsd[40]])
    model.layers[26].set_weights([fnsd[41]])

    model.layers[28].set_weights([fnsd[42], fnsd[43]])
    model.layers[29].set_weights([fnsd[44], fnsd[45]])
    model.layers[30].set_weights([fnsd[46]])
    model.layers[31].set_weights([fnsd[47]])

    model.layers[33].set_weights([fnsd[48], fnsd[49]])
    model.layers[34].set_weights([fnsd[50], fnsd[51]])

    print 'deconv layer finished'
    model.save_weights('fn_sd.h5')
    return model


def flowNetSD_f():
    h = 384#320#384
    w = 512#1152#512
    input = Input(shape=(h, w, 6))
    conv0 =   Conv2D(64,   (3, 3), name = 'conv0',   strides = 1, padding='same', activation=LeakyReLU())(input)
    conv1 =   Conv2D(64,   (3, 3), name = 'conv1',   strides = 2, padding='same', activation=LeakyReLU())(conv0)
    conv1_1 = Conv2D(128,  (3, 3), name = 'conv1_1', strides = 1, padding='same', activation=LeakyReLU())(conv1)
    conv2 =   Conv2D(128,  (3, 3), name = 'conv2',   strides = 2, padding='same', activation=LeakyReLU())(conv1_1)
    conv2_1 = Conv2D(128,  (3, 3), name = 'conv2_1', strides = 1, padding='same', activation=LeakyReLU())(conv2)
    conv3 =   Conv2D(256,  (3, 3), name = 'conv3',   strides = 1, padding='same', activation=LeakyReLU())(conv2_1)
    conv3_1 = Conv2D(256,  (3, 3), name = 'conv3_1', strides = 2, padding='same', activation=LeakyReLU())(conv3)
    conv4 =   Conv2D(512,  (3, 3), name = 'conv4',   strides = 1, padding='same', activation=LeakyReLU())(conv3_1)
    conv4_1 = Conv2D(512,  (3, 3), name = 'conv4_1', strides = 2, padding='same', activation=LeakyReLU())(conv4)
    conv5 =   Conv2D(512,  (3, 3), name = 'conv5',   strides = 1, padding='same', activation=LeakyReLU())(conv4_1)
    conv5_1 = Conv2D(512,  (3, 3), name = 'conv5_1', strides = 2, padding='same', activation=LeakyReLU())(conv5)
    conv6 =   Conv2D(1024, (3, 3), name = 'conv6',   strides = 2, padding='same', activation=LeakyReLU())(conv5_1)
    conv6_1 = Conv2D(1024, (3, 3), name = 'conv6_1', strides = 1, padding='same', activation=LeakyReLU())(conv6)

    predict_flow6 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (conv6_1)
    deconv5 = Conv2DTranspose(512, (4,4), strides = 2, padding = 'same', activation = LeakyReLU(), use_bias = False)(conv6_1)
    up6_5 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow6)

    concat5 = concatenate([conv5_1, deconv5, up6_5], axis=3)
    interconv5 =       Conv2D(512, (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (concat5)
    predict_flow5 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv5)
    deconv4 = Conv2DTranspose(256, (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat5)
    up5_4 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow5)

    concat4 = concatenate([conv4_1, deconv4, up5_4],axis=3)
    interconv4 =       Conv2D(256, (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (concat4)
    predict_flow4 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv4)
    deconv3 = Conv2DTranspose(128, (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat4)
    up4_3 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow4)

    concat3 = concatenate([conv3_1, deconv3, up4_3],axis=3)
    interconv3 =       Conv2D(128, (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (concat3)
    predict_flow3 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv3)
    deconv2 = Conv2DTranspose(64,  (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat3)
    up3_2 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow3)

    concat2 = concatenate([conv2_1, deconv2, up3_2],axis=3)
    interconv2 =       Conv2D(64,  (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (concat2)
    predict_flow2 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv2)

    finalOut = predict_flow2
    #finalOut = tf.image.resize_bilinear(finalOut, tf.stack([h, w]), align_corners=True)
    #finalOut = K.resize_images(finalOut, h, w, "channels_last")
    model = Model(input=input, output=finalOut)
    model.summary()
    return model


if __name__ == '__main__':
    m = flowNetSD_f()
    m = setFNetWeight(m)

    img1 = cv2.imread('img0.ppm')
    img2 = cv2.imread('img1.ppm')
    img = np.concatenate((img1, img2), axis=2)
    img = np.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2]))
    print img.shape
    pred = m.predict(img)*0.05
    print 'im here'

    print pred.shape
    pred = pred[0,:,:,:]

    print pred.shape

    ddd = flow_to_image(pred)
    print ddd.shape
    ddd = cv2.resize(ddd, (512,384))
    cv2.imshow('d', ddd)
    # pred[:,:,0] = preprocessing.normalize(pred[:,:,0])
    # pred[:,:,1] = preprocessing.normalize(pred[:,:,1])
    # cv2.imshow('1', pred[:,:,0])
    # cv2.imshow('2', pred[:,:,1])
    cv2.waitKey(5000)















#end
