
import os
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint, Callback
from keras.optimizers import SGD
import os
import random
import numpy as np
import skimage
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
# from keras import backend as keras
import keras.backend as K
from utilsTwo import DataGenerator
from unetTwo import *
from keras.utils import multi_gpu_model
from mymetrics import *

class ParallerModelCheckPoint(Callback):
    def __init__(self, single_model):
        self.mode_to_save = single_model
        
    def on_epoch_end(self, epoch, logs={}):
        print(r'save model: check'+sname+'/unet-%02d.hdf5'%(epoch+1))
        self.mode_to_save.save_weights(r'check'+sname+'/unet-%02d.hdf5'%(epoch+1))


# 设置使用的显存以及GPU
# 设置可用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5'

# keras设置GPU参数
'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.888
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)
'''
use_gpu=True
gpus=2
batch_size=6
sname='_Twob'
# xception, mobilenetv2
basemodel =  unet(input_size=(None, None, None,2))

# model_file = 'check_mse6/unet-54.hdf5'
# if os.path.exists(model_file):
#     print('loading model:', model_file)
#     basemodel.load_weights(model_file, by_name=True)

if use_gpu:
    parallermodel = multi_gpu_model(basemodel, gpus=gpus)
    checkpoint = ParallerModelCheckPoint(basemodel)
else:
    parallermodel = basemodel
    checkpoint = ModelCheckpoint(r'check'+sname+'/unet-{epoch:02d}.hdf5', save_weights_only=True, verbose=1)
    
# optimizer = SGD(lr=learning_rate, momentum=0.9, clipnorm=5.0)
parallermodel.compile(optimizer = Adam(lr = 1e-4), loss = "binary_crossentropy",#mse binary_crossentropy
                      metrics=['accuracy'])
                                            # metric_precision,
                                            # metric_recall,
                                            # metric_F1score,
                                            # mean_iou_keras])

# parallermodel.compile(optimizer = Adam(lr = 1e-4), 
#        loss = mymse, metrics=['accuracy',mean_iou_keras])#,
#                                             # metric_precision,
#                                             # metric_recall,
#                                             # metric_F1score])

params = {'batch_size':6,
          'dim':(128,128,128),
          'n_channels':2,
          'shuffle': True}
seismPath = "/home/hlsheng/hlsheng_dhale/data/train/sx/"
seisomPath = "/home/hlsheng/hlsheng_dhale/data/train/sxo/"
co2Path = "/home/hlsheng/hlsheng_dhale/data/train/lx/"
seismvPath = "/home/hlsheng/hlsheng_dhale/data/valid/sx/"
seisomvPath = "/home/hlsheng/hlsheng_dhale/data/valid/sxo/"
sembvPath = "/home/hlsheng/hlsheng_dhale/data/valid/lx/"

train_ID=[]
valid_ID=[]
c = 0
for sfile in os.listdir(seismPath):
    if sfile.endswith(".dat") and not sfile.startswith(".DS"):
        if(c<416):
            train_ID.append(sfile)
        # else:
        #     valid_ID.append(sfile)
        c = c+1

cw = 0
for sfile in os.listdir(seismvPath):
    if sfile.endswith(".dat") and not sfile.startswith(".DS"):
        if(cw<104):
            # train_ID.append(sfile)
        # else:
            valid_ID.append(sfile)
        cw = cw+1

train_generator = DataGenerator(dpath=seismPath,fpath=co2Path,spath=seisomPath,
                                data_IDs=train_ID,**params)
valid_generator = DataGenerator(dpath=seismvPath,fpath=sembvPath,spath=seisomvPath,
                                data_IDs=valid_ID,**params)
# train_gen = data_generator(x_train, y_train, batch_size=batch_size)
# test_gen = data_generator(x_test, y_test, batch_size=batch_size)

tensorboard = TensorBoard('log'+sname, write_graph=True)

earlystop = EarlyStopping(monitor='acc', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, 
                                patience=2, min_lr=1e-8)

print('begin training...')
history=parallermodel.fit_generator(train_generator, 
                        steps_per_epoch=c//batch_size, 
                        epochs=100, 
                        verbose=1, 
                        callbacks=[reduce_lr, checkpoint, tensorboard], 
                        validation_data=valid_generator, 
                        validation_steps=cw//batch_size, 
                        initial_epoch=0)
showHistory(history)




