#coding=utf-8
from __future__ import print_function

import itertools
import h5py
import numpy as np
import os
import random
import sys
import cv2
import scipy.misc
import math
from scipy.misc import imresize
from random import shuffle

from keras import backend as K
import theano.tensor as T
import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
import keras.models
import keras.callbacks

from theano.compile.sharedvalue import shared
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.layers import Dense, ZeroPadding2D, Activation#, Dropout,, Flatten, Input, 
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

img_rows = 512
img_cols = 512
nb_epoch = 100 #1000
iteration_size = 100000
mini_batch_size = 10
delta = 16
initial_discount = 0.01
discount_step = 1.0/(100*900)

num_samples_per_epoch = 900#50000 
num_validation_samples = 500#5000

d = shared(initial_discount, name = 'd')

def fcrn_loss(y_true, y_pred):
  loss = K.square(y_pred - y_true)
  
  images = []
  
  for i in range(0, mini_batch_size):
    c_true = y_true[i, 6, :,:].reshape((1, delta, delta))   # The last feature map in the true vals is the 'c' matrix

    c_discounted = T.set_subtensor(c_true[(c_true<=0.0).nonzero()], d.get_value())
    
    final_c = (c_discounted * loss[i,6,:,:])
       
    # Element-wise multiply of the c feature map against all feature maps in the loss
    #final_loss_parts = [(c_true * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)]
    #final_loss_parts.append(final_c)    
    #images.append(K.concatenate(final_loss_parts))

    final_loss_parts_new = np.array([(c_true * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)])
    final_loss_parts_new =final_loss_parts_new*5
    final_loss_parts = list(final_loss_parts_new)
    
    final_loss_parts.append(final_c)
    images.append(K.concatenate(final_loss_parts))   

    tt = K.mean(K.concatenate(images).reshape((mini_batch_size, 7, delta, delta)), axis = 1)
    #mm=K.eval(tt)
    #print (mm)
    #print (mm.shape)
    #print (K.shape(tt)) #Subtensor{::}.0

  return tt
  
class DiscountCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    print("Running callback: " + str(epoch))
    d.set_value(d.get_value() + discount_step)
    
def build_model():
    model = Sequential()
    
    # Layer 1
    model.add(ZeroPadding2D(padding = (2, 2), input_shape=(1, img_rows, img_cols))) #theano (channels,w,h) ,tensorflow (w,h,channels)
    model.add(Convolution2D(64, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 1: " + str(model.layers[-1].output_shape))
    
    # Layer 2
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 2: " + str(model.layers[-1].output_shape))
    
    # Layer 3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 3: " + str(model.layers[-1].output_shape))
 
    # Layer 4
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    # Layer 5
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 5: " + str(model.layers[-1].output_shape))
    
    
    # Layer 6
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 6: " + str(model.layers[-1].output_shape))
    
    
    # Layer 7
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 7: " + str(model.layers[-1].output_shape))
    
    # Layer 8
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    
    # Layer 9
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 9: " + str(model.layers[-1].output_shape))
    

    # Layer 10
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(7, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 10: " + str(model.layers[-1].output_shape))
    
    #sgd = SGD(lr = 10e-4, decay = 5e-4, momentum = 0.9, nesterov = False)
    sgd = SGD(lr = 10e-4, decay = 5e-4, momentum = 0.9, nesterov = False)
    
    model.compile(loss = fcrn_loss, optimizer = sgd, metrics = ['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer = sgd, metrics = ['accuracy'])
    
    return model
        
def batch(iterable, n = 1):
  current_batch = []
  #print ("batch-----------------\n")
  for item in iterable:
    current_batch.append(item)
    #print ("current_batch:{}\n".format(str(current_batch)))
    if len(current_batch) == n:
      yield current_batch
      current_batch = []
      
global a
a = 0

def Fuc():
    global a
    print ("index:{}\n".format(a))
    a = a + 1
   
def exemplar_generator(db_iters, batch_size):
  while True:
    #print ("+++++++++++++++++\n")
    #print ("db_iters:{}\n".format(db_iters))    
    db_path = "../small_data"
    dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
    db_iters = map(lambda x: load_db(x), dbs)
    
    for chunk in batch(itertools.chain.from_iterable(db_iters), batch_size):
      X = []
      Y = []
           
      for item in chunk:
        X.append(item[:].reshape(1, img_rows, img_cols))
        labels = np.array(item.attrs['label']).transpose(2, 0, 1)
        Y.append(labels.reshape(7, delta, delta))  
        #Fuc()
      yield (np.array(X), np.array(Y))
      
def load_db(db_filename):
  try:
    db = h5py.File(db_filename, 'r')
    return db['data'].itervalues()
  except:
    print(sys.exc_info()[1])
    return []

def get_move_roi(tl,tr,bl,br, center0,center1):
    dx = center1[0] - center0[0]
    dy = center1[1] - center0[1]
      
    tl = (tl[0]-dx, tl[1]-dy)
    tr = (tr[0]-dx, tr[1]-dy)
    bl = (bl[0]-dx, bl[1]-dy)
    br = (br[0]-dx, br[1]-dy)
    return (tl,tr,bl,br)
    
    
def get_roi_center(tl,tr,bl,br):
    x_min = min(tl[0],tr[0],bl[0],br[0])
    x_max = max(tl[0],tr[0],bl[0],br[0])
    y_min = min(tl[1],tr[1],bl[1],br[1])
    y_max = max(tl[1],tr[1],bl[1],br[1])
    return (float((x_min + x_max))/2.0, float((y_min + y_max))/2.0)
 
def get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    x0_min = min(tl0[0],tr0[0],bl0[0],br0[0])
    x0_max = max(tl0[0],tr0[0],bl0[0],br0[0])
    y0_min = min(tl0[1],tr0[1],bl0[1],br0[1])
    y0_max = max(tl0[1],tr0[1],bl0[1],br0[1])
    
    x1_min = min(tl1[0],tr1[0],bl1[0],br1[0])
    x1_max = max(tl1[0],tr1[0],bl1[0],br1[0])
    y1_min = min(tl1[1],tr1[1],bl1[1],br1[1])
    y1_max = max(tl1[1],tr1[1],bl1[1],br1[1])
    
    x_min = min(x0_min,x1_min)
    x_max = max(x0_max,x1_max)
    y_min = min(y0_min,y1_min)
    y_max = max(y0_max,y1_max)
    return (x_min,x_max,y_min,y_max)
    
def get_rotate_point(tl,tr,bl,br,cos,sin):
    tl = (tl[0]*cos-tl[1]*sin, tl[0]*sin+tl[1]*cos)
    tr = (tr[0]*cos-tr[1]*sin, tr[0]*sin+tr[1]*cos)
    bl = (bl[0]*cos-bl[1]*sin, bl[0]*sin+bl[1]*cos)
    br = (br[0]*cos-br[1]*sin, br[0]*sin+br[1]*cos)
    '''print ("tl:{}\n ".format(tl))
    R = math.pow(cos,2) + math.pow(sin,2)
    R = math.sqrt(R)
    print ("R:{}\n ".format(R))
    print ("cos:{}\n".format(cos))
    print ("sin:{}\n".format(sin))'''
    return (tl,tr,bl,br)

def get_cross(pt1,pt2,pt):
    return (pt2[0]-pt1[0])*(pt[1]-pt1[1]) - (pt[0]-pt1[0])*(pt2[1]-pt1[1])

def IsPointInMatrix(tl,tr,bl,br,pt):
    return get_cross(tl,bl,pt) * get_cross(br,tr,pt) >= 0 and get_cross(bl,br,pt) * get_cross(tr,tl,pt) >= 0
 
def get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    (x_min,x_max,y_min,y_max) = get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
    union_num = 0
    inter_num = 0
    
    for col in range(x_min,x_max):
        for row in range(y_min,y_max):
            pt = (col,row)
            if IsPointInMatrix(tl0,tr0,bl0,br0,pt) and IsPointInMatrix(tl1,tr1,bl1,br1,pt):
                inter_num = intersection_num + 1
            elif (not IsPointInMatrix(tl0,tr0,bl0,br0,pt)) or (not IsPointInMatrix(tl1,tr1,bl1,br1,pt)):
                union_num = union_num + 1
    return float(inter_num)/union_num
                          
def nms(lst,confideres,threshold):
    if len(lst)==0:
        return []
       
    I = np.argsort(confideres)
    pick = np.zeros_like(confideres, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        tl0 = lst[i][0]
        tr0 = lst[i][1]
        bl0 = lst[i][2]
        br0 = lst[i][3]
        
        tl1 = lst[idx][0]
        tr1 = lst[idx][1]
        bl1 = lst[idx][2]
        br1 = lst[idx][3]      
        o = get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick
     
 
def draw_roi(img,res):
    '''print (len(res))
    print (len(res[0]))
    print (len(res[0][0]))
    print (len(res[0][0][0]))

    for i in range(0,len(res[0])):
       print (res[0][i][0][0])#(res[0][i])
       print ('\n')'''
    print ("..........")
    x = res[0,0,:,:]
    y = res[0,1,:,:]
    w = res[0,2,:,:]
    h = res[0,3,:,:]
    cos = res[0,4,:,:]
    sin = res[0,5,:,:]
    c = res[0,6,:,:]
    
    roi = []
    confideres = []
    img_src = img.copy()
    for row in range(0,16):
        for col in range(0,16):
            if w[row][col]>0.0 and h[row][col]>0.0 and c[row][col] > 0.5:
                centerX = x[row][col]*32 + col
                centerY = y[row][col]*32 + row
                ww = 32*math.exp(w[row][col])
                hh = 32*math.exp(h[row][col])
                
                tl = (centerX-ww/2, centerY-hh/2)
                tr = (centerX+ww/2, centerY-hh/2)
                bl = (centerX-ww/2, centerY+hh/2)
                br = (centerX+ww/2, centerY+hh/2)                
                center0 = get_roi_center(tl,tr,bl,br)
                
                cv2.line(img_src,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                cv2.line(img_src,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                cv2.line(img_src,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                cv2.line(img_src,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                
                print (tl)
                (tl,tr,bl,br) = get_rotate_point(tl,tr,bl,br,cos[row][col],sin[row][col])
                center1 = get_roi_center(tl,tr,bl,br)           
                (tl,tr,bl,br) = get_move_roi(tl,tr,bl,br, center0,center1)
                
                roi.append([tl,tr,bl,br])
                confideres.append(c[row][col]) 
                #print (tl)
                cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                #cv2.rectangle(img, (xx,yy), (int(xx+w[row][col]*512),int(yy+h[row][col]*512)),(0,0,255),1)
                #print (x[row][col],y[row][col])
                print (ww,hh,c[row][col])
    cv2.imwrite("test.bmp", img)
    cv2.imwrite("img_src.bmp", img_src)
    return (roi,confideres)

def load_exemplars(db_path):
  dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
  print ("load_exemplars ...........")
  return exemplar_generator(map(lambda x: load_db(x), dbs), mini_batch_size)

if __name__ == '__main__':
  model_file = "bb-fcrn-model_weight_pr"
  train_db_path = "../h5"    #"/path/to/dbs"
  validate_db_path = "../h5" #"/path/to/dbs"
  
  print("Loading data...")
  
  train = load_exemplars(train_db_path)
  validate = load_exemplars(validate_db_path)
  
  print("Data loaded.")
  print("Building model...")
  
  model = build_model()
  
  checkpoint = keras.callbacks.ModelCheckpoint(model_file + ".h5",
                                               monitor = "acc",
                                               verbose = 1,
                                               save_best_only = True,
                                               save_weights_only = True,
                                               mode = 'auto')
  
  earlystopping = keras.callbacks.EarlyStopping(monitor = 'loss',
                                                min_delta = 0,
                                                patience = 5,
                                                verbose = 1,
                                                mode = 'auto')
  
  discount = DiscountCallback()
  
  csvlogger = keras.callbacks.CSVLogger(model_file + "-log.csv", append = True)
  
      
  if os.path.exists(model_file + ".h5"):
    model.load_weights(model_file + ".h5")
    print ("load weights ok!")
    image_path = "7.jpg"
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    if img == None:
        os._exit() 
    
    h_scale = img_rows / float(img.shape[0])
    w_scale = img_cols / float(img.shape[1])
          
    img_color = imresize(img, (int(img_rows), int(img_cols)), interp = 'bicubic')
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    img1 = np.expand_dims(img, axis=0) #扩一维
    img1 = np.expand_dims(img1, axis=0)
    #print (img.shape)    
        
    res = model.predict(img1) #predict_on_batch(np.array(train.next()[0])) #1*7*16*16 (batch_size*7*16*16)
    draw_roi(img_color, res)
    
  else:
    model.fit_generator(train,
                      samples_per_epoch = num_samples_per_epoch,
                      nb_epoch = nb_epoch,
                      verbose = 1,
                      validation_data = validate,
                      nb_val_samples = num_validation_samples,
                      max_q_size = 10,
                      pickle_safe = True,
                      callbacks = [checkpoint, earlystopping, csvlogger, discount])
                      
                     
