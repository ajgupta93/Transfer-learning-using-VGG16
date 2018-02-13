#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:47:33 2017

@author: nimish
"""

import os
from keras.preprocessing import image
import numpy as np

os.system('ulimit -s unlimited')

def read_image(filename, target_size):
	img = image.load_img(filename, target_size=target_size)
	img = image.img_to_array(img)
	return img.reshape((1,)+img.shape)

def concatenate_images(images, labels, img, label):
	if images is not None:
		images = np.concatenate((images, img))
		labels = np.concatenate((labels, label))
	else:
		images = img
		labels = label

	return images, labels

def read_cal256(path,target_size,trainSize, testSize):
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    unique_labels = []
    for dirName, subDirs, files in os.walk(path):
        for label in subDirs:
            if(unique_labels==None or label not in unique_labels and '.' in label and label != 'clutter'):
                    unique_labels.append(label)  
    
    perClassCount = np.zeros((1,len(unique_labels)))
    perClassCountTest = np.zeros((1,len(unique_labels)))
    
    
    #for dirName, subDirs, files in os.walk(path):
    i=0
    for subDir in unique_labels:
        print i
        i +=1
        dirDone = False
        for dirName, sdir, files in os.walk(os.path.join(path,subDir)):
            np.random.shuffle(files)
            for filename in files:
                if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
                    label = subDir
                    lblIdx = unique_labels.index(label)
                    
                    if(perClassCount[0][lblIdx] < trainSize):
                        perClassCount[0][lblIdx] += 1
                        img = read_image(os.path.join(dirName,filename), target_size)
                        enc = np.zeros((1,len(unique_labels)))
                        enc[0][lblIdx] = 1
                        train_images, train_labels = concatenate_images(train_images, train_labels, img, enc)
                        
                    elif(perClassCountTest[0][lblIdx] < testSize):
                        perClassCountTest[0][lblIdx] += 1
                        img = read_image(os.path.join(dirName,filename), target_size)
                        enc = np.zeros((1,len(unique_labels)))
                        enc[0][lblIdx] = 1
                        test_images, test_labels = concatenate_images(test_images, test_labels, img, enc)
                if(perClassCount[0][lblIdx]==trainSize and perClassCountTest[0][lblIdx]==testSize):
                    dirDone = True
                    break
            if dirDone == True:
                break
    
    return train_images, train_labels, test_images, test_labels, unique_labels

def read_UT(path,target_size,trainSize, testSize):
    train_images = None
    train_labels = None
    test_images = None
    test_labels = None
    unique_labels = []
    for dirName, subDirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
                label = filename.partition('_')[0]
                if(unique_labels==None or label not in unique_labels):
                    unique_labels.append(label)
    
    perClassCount = np.zeros((1,len(unique_labels)))
    perClassCountTest = np.zeros((1,len(unique_labels)))
    
    
    for dirName, subDirs, files in os.walk(path):
        np.random.shuffle(files)
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
                label = filename.partition('_')[0]
                lblIdx = unique_labels.index(label)
                
                if(perClassCount[0][lblIdx] < trainSize):
                    perClassCount[0][lblIdx] += 1
                    img = read_image(os.path.join(dirName,filename), target_size)
                    enc = np.zeros((1,len(unique_labels)))
                    enc[0][lblIdx] = 1
                    train_images, train_labels = concatenate_images(train_images, train_labels, img, enc)
                    
                elif(perClassCountTest[0][lblIdx] < testSize):
                    perClassCountTest[0][lblIdx] += 1
                    img = read_image(os.path.join(dirName,filename), target_size)
                    enc = np.zeros((1,len(unique_labels)))
                    enc[0][lblIdx] = 1
                    test_images, test_labels = concatenate_images(test_images, test_labels, img, enc)
                
    
    return train_images, train_labels, test_images, test_labels, unique_labels

if __name__ == "__main__":
    target_size=(224, 224)
    train_images, train_labels, test_images, test_labels, oneHotDecode = read_UT('./pictures_all', target_size,16,4)
    np.save('./dataUT/train_images.npy', train_images)
    np.save('./dataUT/train_labels.npy', train_labels)
    np.save('./dataUT/test_images.npy', test_images)
    np.save('./dataUT/test_labels.npy', test_labels)
    np.save('./dataUT/label_decode.npy', oneHotDecode)
    print 'urban tribes ready to load'
    del train_images, train_labels, test_images, test_labels, oneHotDecode
    
    train_images, train_labels, test_images, test_labels, oneHotDecode = read_cal256('./256_ObjectCategories', target_size,16,4)
    np.save('./data256/train_images.npy', train_images)
    np.save('./data256/train_labels.npy', train_labels)
    np.save('./data256/test_images.npy', test_images)
    np.save('./data256/test_labels.npy', test_labels)
    np.save('./data256/label_decode.npy', oneHotDecode)
    print 'cal256 ready to load'
    del train_images, train_labels, test_images, test_labels, oneHotDecode