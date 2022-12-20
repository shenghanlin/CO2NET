import os
import sys
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import load_model
#from unetd import mybce
import matplotlib.pyplot as plt
from scipy.interpolate import interpn
from scipy import interpolate
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from unet import *
datapath = '../../'
def main(argv):
  loadModel(argv[1])
  gop10()
  # gop07_01()
  # gop07_04()
  # gop07_06()
  # gop08()
  # gop99()


  
def loadModel(mk):
  global model
  # model = load_model('check/vpik-'+str(mk)+'.hdf5',custom_objects={'tf_total_loss': tf_total_loss})
  model = unet(input_size=(None, None, None,1))
  model.load_weights('./check_mse2/unet-'+str(mk)+'.hdf5')#mse 95 bin 107
  #model.load_weights('./check/unet-22.hdf5')
  #model.load_weights('./check/unet-22.hdf5', by_name=True)
  # model = load_model('../deeplearning/net/check/fseg-'+str(mk)+'.hdf5')
  

def gop99(): 

  fpath = "./partcutnea/"
  fname ="99p01cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")


def gop07_01(): 

  fpath = "./partcutnea/"
  fname ="01p07cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")

def gop07_04(): 

  fpath = "./partcutnea/"
  fname ="04p07cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")


def gop07_06(): 

  fpath = "./partcutnea/"
  fname ="06p07cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")

def gop08(): 

  fpath = "./partcutnea/"
  fname ="08p08cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")

def gop10(): 

  fpath = "./partcutnea/"
  fname ="10p10cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  # mask = np.ones_like(gx)
  # print(mask.shape)
  # for i in range(256):
  #   maskw = 1+i*4.5/256
  #   mask[i,:,:] = maskw * mask[i,:,:]
  # gx = gx*mask
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")


def gop10(): 

  fpath = "./partcutnea/"
  fname ="10p10cutnea.dat"
  n1,n2,n3 = 256,468,245
  gx = np.fromfile(fpath+fname,dtype=np.single)# dtype='>f'
  print(np.max(gx),np.min(gx))
  gx = np.reshape(gx,(n3,n2,n1))
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.transpose(gx)
  mask = np.ones_like(gx)
  print(mask.shape)
  for i in range(256):
    maskw = 1+i*4.5/256
    mask[i,:,:] = maskw * mask[i,:,:]
  gx = gx*mask
  m1 = int(np.ceil(n1/16)*16)
  m2 = int(np.ceil(n2/16)*16)
  m3 = int(np.ceil(n3/16)*16)
  m1,m2,m3=128,512,256  #m1,m2,m3=256,512,256
  
  fx = goPredict(m1,m2,m3,gx)#make a prediction 
  fx = np.transpose(fx)
  print(np.min(fx),np.max(fx))
  fx.tofile(fpath+"fp_"+'mseN_'+fname,format="%4") #  fx.tofile(fpath+"fpd.dat",format="%4")


#m1,m2,m3:the dimensions of a subset
#each needs be divisible by 16,
#choose large dimensions if your CPU/GPU memory allows
def pick(fx):
  a = np.argmax(fx, axis=-1)
  a = a*0.02+1.0
  print(a)
  return a

def section(fx):
  a = np.max(fx, axis=1)
  # a = a*0.02+1.0
  print(a.shape)
  return a

def goPredict(m1,m2,m3,gx):
  n1,n2,n3=gx.shape 
  if m1>=n1 and m2>=n2 and m3>=n3:
    return goPredictFull(m1,m2,m3,gx)
  else:
    return goPredictSubs(m1,m2,m3,gx)

def goPredictFull(m1,m2,m3,gx): 
  n1,n2,n3=gx.shape 
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs

  fx = np.zeros((n1,n2,n3),dtype=np.single)
  gk = np.zeros((1,m1,m2,m3,1),dtype=np.single)
  gk[0,:n1,:n2,:n3,0] = gx[:,:,:]
  fk = model.predict(gk,verbose=1) #fault prediction
  fx[:,:,:] = fk[0,:n1,:n2,:n3,0]
  #set the bounds
  fx[-1,:,:]=fx[-2,:,:]
  fx[:,-1,:]=fx[:,-2,:]
  fx[:,:,-1]=fx[:,:,-2]
  return fx

#m1,m2,m3:the dimensions of a subset
#each needs be divisible by 16,
#choose large dimensions if your CPU/GPU memory allows
def goPredictSubs(m1,m2,m3,gx): 
  n1,n2,n3=gx.shape 
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx-gm
  gx = gx/gs
  p1,p2,p3=120,120,120 #overlap
  fx = np.zeros((n1,n2,n3),dtype=np.single)
  c1=1+int(np.ceil(float(n1-m1)/(m1-p1)))
  c2=1+int(np.ceil(float(n2-m2)/(m2-p2)))
  c3=1+int(np.ceil(float(n3-m3)/(m3-p3)))
  for k3 in range(c3):
    for k2 in range(c2):
      for k1 in range(c1):
        gk = np.zeros((m1,m2,m3),dtype=np.single)
        b1,b2,b3 = k1*(m1-p1),k2*(m2-p2),k3*(m3-p3)
        e1,e2,e3 = b1+m1,b2+m2,b3+m3
        e1 = min(e1,n1)
        e2 = min(e2,n2)
        e3 = min(e3,n3)
        d1,d2,d3 = e1-b1,e2-b2,e3-b3
        gk[0:d1,0:d2,0:d3] = gx[b1:e1,b2:e2,b3:e3]
        gm,gs = np.mean(gk),np.std(gk)
        gk = gk-gm
        gk = gk/gs
        gk = np.reshape(gk,(1,m1,m2,m3,1))
        fk = model.predict(gk,verbose=1) #fault prediction
        t1 = min(int(p1/2),b1)
        t2 = min(int(p2/2),b2)
        t3 = min(int(p3/2),b3)
        fx[b1+t1:e1,b2+t2:e2,b3+t3:e3] = fk[0,t1:d1,t2:d2,t3:d3,0]
  #set the bounds
  fx[-1,:,:]=fx[-2,:,:]
  fx[:,-1,:]=fx[:,-2,:]
  fx[:,:,-1]=fx[:,:,-2]
  return fx


def loadData(n1,n2,n3,path,fname):
  gx = np.fromfile(str(path)+str(fname),dtype=np.single)
  #gmin,gmax=np.min(gx)/5,np.max(gx)/5
  gm,gs = np.mean(gx),np.std(gx)
  gx = gx-gm
  gx = gx/gs
  gx = np.reshape(gx,(n3,n2,n1))
  # gx = np.transpose(gx)
  return gx

def loadDatax(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def pick_inter(fx,high):
  a = np.argmax(fx, axis=-1)
  t = []
  v = []
  for i,energy in enumerate(a):
    if fx[i][energy]>=high:
      t.append(i)
      # v.append(energy*0.02+1.0)
      v.append(energy)

  A = np.polyfit(t,v,5)
  Ax = np.poly1d(A) 
  x = np.arange(1, 1024, 1)
  y_new=Ax(x)
  print(y_new)
  # x_new = np.linspace(1,1024,1)
  # y_new =li(x_new)#给出x、y取值范围#
  return y_new


def goPredict2D(gx): 
  fk = model.predict(gx,verbose=1) #fault prediction
  return fk

def loadData2D(n1,n2,path,fname,norm=True):
    fname = fname+'.dat'
    gx = np.fromfile(path+fname,dtype=np.single)
    np.reshape(gx,(n1,n2))
    #gmin,gmax=np.min(gx)/5,np.max(gx)/5
    if norm==True:
        gm,gs = np.mean(gx),np.std(gx)
        gx = gx-gm
        gx = gx/gs
    #gx = np.reshape(gx,(n2,n1))
    gx = np.transpose(gx)
    gx = np.reshape(gx, (1,n1,n2,1))
    return gx

def loadDatax(n1,n2,n3,path,fname):
  gx = np.fromfile(path+fname,dtype=np.single)
  gx = np.reshape(gx,(n3,n2,n1))
  gx = np.transpose(gx)
  return gx

def sigmoid(x):
    s=1.0/(1.0+np.exp(-x))
    return s

def plot2d(gx,fx,fp,at=1,png=None):
  fig = plt.figure(figsize=(15,5))
  #fig = plt.figure()
  ax = fig.add_subplot(131)
  ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(132)
  ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ax = fig.add_subplot(133)
  ax.imshow(fp,vmin=0,vmax=1.0,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  if png:
    plt.savefig(pngDir+png+'.png')
  #cbar = plt.colorbar()
  #cbar.set_label('Fault probability')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
    main(sys.argv)


