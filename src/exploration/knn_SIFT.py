# knn_SIFT.py 
# Monolithic script to evaluate the KNN classifier with descriptors generated from SIFT 

from sklearn.model_selection import train_test_split,cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import glob
import random


# SIFT obtains & returns image descriptors
def SIFT(imgs): 
    sift = cv2.SIFT_create() 
    descriptors = []

    for img in imgs: 
        _,des = sift.detectAndCompute(img,None)
        descriptors.append(des)
    
    return descriptors
    
# Using K-means clustering 
# 
def cluster(): #TODO 
    pass


# Bag of Visual Words Quantization. 
# Generates the feature space 
def BOVW(descriptors,numSamples,clusters): 
    pass #TODO


# Cross-validated KNN using sklearn. 
def crossValidateKNN(train_X,train_Y,test_X,test_Y,folds=10):
    pass #TODO 

