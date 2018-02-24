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
    h = 320#384
    w = 1152#512
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
    interconv5 =       Conv2D(512, (3,3), strides = 1, padding = 'same', activation=LeakyReLU(),      use_bias = True) (concat5)
    predict_flow5 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv5)
    deconv4 = Conv2DTranspose(256, (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat5)
    up5_4 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow5)

    concat4 = concatenate([conv4_1, deconv4, up5_4],axis=3)
    interconv4 =       Conv2D(256, (3,3), strides = 1, padding = 'same', activation=LeakyReLU(),      use_bias = True) (concat4)
    predict_flow4 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv4)
    deconv3 = Conv2DTranspose(128, (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat4)
    up4_3 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow4)

    concat3 = concatenate([conv3_1, deconv3, up4_3],axis=3)
    interconv3 =       Conv2D(128, (3,3), strides = 1, padding = 'same', activation=LeakyReLU(),      use_bias = True) (concat3)
    predict_flow3 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv3)
    deconv2 = Conv2DTranspose(64,  (4,4), strides = 2, padding = 'same', activation=LeakyReLU(),   use_bias = False)(concat3)
    up3_2 =   Conv2DTranspose(2,   (4,4), strides = 2, padding = 'same', activation='linear',      use_bias = False)(predict_flow3)

    concat2 = concatenate([conv2_1, deconv2, up3_2],axis=3)
    interconv2 =       Conv2D(64,  (3,3), strides = 1, padding = 'same', activation=LeakyReLU(),      use_bias = True) (concat2)
    predict_flow2 =    Conv2D(2,   (3,3), strides = 1, padding = 'same', activation='linear',      use_bias = True) (interconv2)

    finalOut = [predict_flow2, predict_flow3, predict_flow4, predict_flow5, predict_flow6]
    #finalOut = tf.image.resize_bilinear(finalOut, tf.stack([h, w]), align_corners=True)
    #finalOut = K.resize_images(finalOut, h, w, "channels_last")
    model = Model(input=input, output=finalOut)
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    m = flowNetSD_f()
    #m = setFNetWeight(m)

    img1 = cv2.imread('932.png')
    img2 = cv2.imread('933.png')
    img = np.concatenate((img1, img2), axis=2)
    img = np.reshape(img, (1,img.shape[0],img.shape[1],img.shape[2]))
    print img.shape

    img1g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1g, img2g, None, 0.5, 3, 10, 1, 1, 1.2, 0)
    print flow.shape
    flow2 = cv2.resize(flow, (80,288))
    flow2 = np.reshape(flow2, (1, flow2.shape[1], flow2.shape[0], flow2.shape[2]))

    flow3 = cv2.resize(flow, (40,144))
    flow3 = np.reshape(flow3, (1, flow3.shape[1], flow3.shape[0], flow3.shape[2]))

    flow4 = cv2.resize(flow, (20,72))
    flow4 = np.reshape(flow4, (1, flow4.shape[1], flow4.shape[0], flow4.shape[2]))

    flow5 = cv2.resize(flow, (10,36))
    flow5 = np.reshape(flow5, (1, flow5.shape[1], flow5.shape[0], flow5.shape[2]))

    flow6 = cv2.resize(flow, (5,18))
    flow6 = np.reshape(flow6, (1, flow6.shape[1], flow6.shape[0], flow6.shape[2]))

    m.load_weights('temp2.h5')
    #m.fit(img, [flow2, flow3, flow4, flow5, flow6], verbose= 1, epochs=1000)
    #m.save_weights('temp2.h5')
    pred = m.predict(img)[0]
    print 'im here'

    print pred.shape
    pred = pred[0,:,:,:]
    print pred.shape

    diff = flow2-pred
    print np.mean(diff)
    print np.max(flow2)


    ddd = flow_to_image(pred)
    print ddd.shape
    #ddd = cv2.resize(ddd, (1152,320))
    cv2.imshow('d', ddd)
    cv2.waitKey(10000)















#end
