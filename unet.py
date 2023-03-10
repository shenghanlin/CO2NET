# Simplified unet for fault segmentation
# The original u-net architecture is more complicated than necessary 
# for our task of fault segmentation.
# We significanlty reduce the number of layers and features at each 
# layer to save GPU memory and computation but still preserve high 
# performace in fault segmentation.

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import matplotlib.pyplot as plt
import keras.backend as K

def unet(pretrained_weights = None,input_size = (None,None,None,1)):
    nf1 = 32
    nf2 = nf1*2
    nf3 = nf2*2
    nf4 = nf3*2
    nf5 = nf4*2
    inputs = Input(input_size)
    conv1 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(inputs)
    conv1 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool1)
    conv2 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool2)
    conv3 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool3)
    conv4 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)

    conv5 = Conv3D(nf5, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool4)
    conv5 = Conv3D(nf5, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv4], axis=-1)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up6)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv3], axis=-1)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up7)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2,2,2))(conv7), conv2], axis=-1)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up8)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2,2,2))(conv8), conv1], axis=-1)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up9)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv9)

    conv10 = Conv3D(1, (1,1,1))(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    #model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy_balanced, metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-4), loss = "mse", metrics = ['accuracy'])
  
    return model



def mybce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)


def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score  

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  #'val_loss', 'val_accuracy', 'val_metric_precision', 'val_metric_recall', 'val_metric_F1score', 
  # 'loss', 'accuracy', 'metric_precision', 'metric_recall', 'metric_F1score', 'lr'
  if history.history['accuracy']:
    fig = plt.figure(figsize=(10,6))
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("acc.png")
#   plt.show()

  # summarize history for loss
  if history.history['loss']:
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.savefig("loss.png")

#   plt.show()

#   if history.history['metric_recall']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_recall'])
#     plt.plot(history.history['val_metric_recall'])
#     plt.title('Model metric recall',fontsize=20)
#     plt.ylabel('metric recall',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("recall.png")

# #   plt.show()
#   if history.history['metric_F1score']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_F1score'])
#     plt.plot(history.history['val_metric_F1score'])
#     plt.title('Model metric_F1score',fontsize=20)
#     plt.ylabel('metric_F1score',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("F1.png")
# #   plt.show()
#   if history.history['metric_precision']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['metric_precision'])
#     plt.plot(history.history['val_metric_precision'])
#     plt.title('Model metric_precision',fontsize=20)
#     plt.ylabel('metric_precision',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("preci.png")

#   if history.history['mean_iou_keras']:
#     fig = plt.figure(figsize=(10,6))
#     plt.plot(history.history['mean_iou_keras'])
#     plt.plot(history.history['val_mean_iou_keras'])
#     plt.title('Model mean_iou_keras',fontsize=20)
#     plt.ylabel('mean_iou_keras',fontsize=20)
#     plt.xlabel('Epoch',fontsize=20)
#     plt.legend(['train', 'test'], loc='center right',fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.tick_params(axis='both', which='minor', labelsize=18)
#     plt.savefig("mean_iou_keras.png")

# #   plt.show()
