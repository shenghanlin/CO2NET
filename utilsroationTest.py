import numpy as np
import keras
import random
from keras.utils import to_categorical
import random

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(128,128,128), dim2=(128,128,128), 
             n_channels=1, shuffle=True):
    'Initialization'
    self.dim   = dim
    self.dim2   = dim2
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    a = 1 #data augumentation
    X = np.zeros((a*self.batch_size, *self.dim2, self.n_channels),dtype=np.single)
    Y = np.zeros((a*self.batch_size, *self.dim2, self.n_channels),dtype=np.single)
    for k in range(self.batch_size):
      gx  = np.fromfile(self.dpath+data_IDs_temp[k],dtype=np.single)
      fx  = np.fromfile(self.fpath+data_IDs_temp[k],dtype=np.single)
      gx = np.reshape(gx,(self.dim[2],self.dim[1],self.dim[0]))
      fx = np.reshape(fx,(self.dim[2],self.dim[1],self.dim[0]))
      fx = 4*np.clip(fx,0,1) 
      # fx = np.clip(fx,0,1)
      
      # di1,di2,di3 = random.randint(0,self.dim[2]-128),random.randint(0,self.dim[1]-128),random.randint(0,self.dim[0]-128)
      # gx = gx[di1:di1+128,di2:di2+128,di3:di3+128]
      # fx = fx[di1:di1+128,di2:di2+128,di3:di3+128]



      gm = np.mean(gx)
      gs = np.std(gx)
      gx = gx-gm
      gx = gx/gs
      gx = np.transpose(gx)
      fx = np.transpose(fx)
      c = k*a
      X[c+0,] = np.reshape(gx, (*self.dim2,self.n_channels))
      Y[c+0,] = np.reshape(fx, (*self.dim2,self.n_channels))  
      #X[c+0,] = np.reshape(np.flipud(gx), (*self.dim,self.n_channels))  
      #Y[c+0,] = np.reshape(np.flipud(fx), (*self.dim,self.n_channels))  
      # i = random.randint(0,3)
      # X[c+1,] = np.reshape(np.rot90(gx,i,(2,1)), (*self.dim2,self.n_channels))
      # Y[c+1,] = np.reshape(np.rot90(fx,i,(2,1)), (*self.dim2,self.n_channels))  
      '''
      for i in range(1,4):
        X[c+i+1,] = np.reshape(np.rot90(gx,i,(2,1)), (*self.dim,self.n_channels))
        Y[c+i+1,] = np.reshape(np.rot90(fx,i,(2,1)), (*self.dim,self.n_channels))  
      '''
    return X,Y
