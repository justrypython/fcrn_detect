#encoding:UTF-8

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
import matplotlib.pyplot as plt

mini_batch_size = 10
img_rows = 512
img_cols = 512
delta = 16
    
def get_squareBoxes(res): 
    x = res[0,0,:,:]
    y = res[0,1,:,:]
    w = res[0,2,:,:]
    h = res[0,3,:,:]
    cos = res[0,4,:,:]
    sin = res[0,5,:,:]
    c = res[0,6,:,:]
    
    boxes = []
    confideres = []
    for row in range(0,16):
        for col in range(0,16):
            #if w[row][col]>0.0 and h[row][col]>0.0 and c[row][col] > 0.00001:
            if c[col][row] > 0.00001:
                centerX = x[col][row]*32 + col*32+16
                centerY = y[col][row]*32 + row*32+16
                ww = 32*math.exp(w[col][row])
                hh = 32*math.exp(h[col][row])
                cc = cos[col][row]
                ss = sin[col][row]
                
                tl = (centerX-(ww/2*cc-hh/2*ss), 
                      centerY-(ww/2*ss+hh/2*cc))
                tr = (centerX+(ww/2*cc+hh/2*ss), 
                      centerY+(ww/2*ss-hh/2*cc))
                bl = (centerX-(ww/2*cc+hh/2*ss), 
                      centerY-(ww/2*ss-hh/2*cc))
                br = (centerX-(hh/2*ss-ww/2*cc), 
                      centerY+(hh/2*cc+ww/2*ss))                
                
                print (ww,hh, c[col][row])
                #if ww>20 and hh>20:
                boxes.append([tl,tr,bl,br])
                confideres.append(c[col][row]) 
                          
    return (boxes,confideres)   


def draw_squareBoxes(img, boxes, confideres, threshold): 
    print ("final boxes num:{}\n".format(len(boxes)))
    if len(boxes) < 1:
        return

    #final_boxes = nms_square(boxes, threshold, 'Union')    
    for i in range(len(boxes)):
        [tl,tr,bl,br] = boxes[i][:]
        if confideres[i] > 0.0001:#0.4:
            #plt.imshow(img)
            #plt.show()
            s = str(round(confideres[i],2))
            cv2.putText(img, s, (int((tl[0]+br[0])/2),int((tl[1]+br[1])/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255 ,0), thickness = 1, lineType = 8) 
            #cv2.rectangle(img,(int(tl[0]),int(tl[1])),(int(br[0]),int(br[1])),(255,255,255),1)        
            cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(255,255,255),1)
            cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(255,255,255),1)
            cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(255,255,255),1)
            cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(255,255,255),1)            
    cv2.imwrite("result.bmp", img)  
    
        
def batch(iterable, n = 1):
    current_batch = []
    #print ("batch-----------------\n")
    for item in iterable:
        current_batch.append(item)
        #print ("current_batch:{}\n".format(str(current_batch)))
        if len(current_batch) == n:
            yield current_batch
            current_batch = []  
            

def exemplar_generator(db_iters, batch_size):
    while True:
        #print ("+++++++++++++++++\n")
        #print ("db_iters:{}\n".format(db_iters))    
        db_path = "../small_data"
        dbs = map(lambda x: "../small_data" + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
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
        return db['data'].values()
    except:
        print(sys.exc_info()[1])
        return []
    

def load_exemplars(db_path):
    dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
    print ("load_exemplars ...........")
    return exemplar_generator(map(lambda x: load_db(x), dbs), mini_batch_size)


if __name__ == '__main__':

    train_db_path = "../small_data" #"/path/to/dbs"
    print("Loading data...")
    train = load_exemplars(train_db_path)
    x, y = train.next()
    img = x[6, 0]
    res = y[6].reshape((1, )+y[6].shape)
    [boxes, confideres] = get_squareBoxes(res)
    draw_squareBoxes(img, boxes, confideres, 0.4)