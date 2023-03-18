# knn_SIFT.py 
# Monolithic script to evaluate the KNN classifier with descriptors generated from SIFT 

from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import numpy as np 
import pandas as pd
import cv2
import glob


# SIFT obtains & returns image descriptors
def SIFT(imgs): 
    sift = cv2.SIFT_create() 
    descriptors = []
    keypoints = []

    for img in imgs: 
        kps,des = sift.detectAndCompute(img,None)
        descriptors.append(des)
        keypoints.append(kps)

    return keypoints,descriptors
    

# Using K-Means clustering for feature reduction. 
# Optimal K is determined by elbow method (see elbow_kmeans.py)
def cluster(descriptors,k = 60): #TODO 
    clusters = KMeans(k,random_state=0).fit(descriptors)
    return clusters 

# Data binning through normalized histograms. 
def binData(keypoints,descriptors,clusters):
    hists = []
    for kps,des in zip(keypoints,descriptors):
        hist = np.zeros(len(clusters.labels_))
        normFact = np.size(kps)
        for d in des: 
            bin = clusters.predict(list(d))
            hist[bin] += 1/normFact
        hists.append(hist)
    return hists

# Cross-validated KNN using sklearn. 
def crossValidateKNN(train_X,train_Y,test_X,test_Y,folds=10):
    
    pass #TODO 


# Evaluate classifier & display confusion matrix
def evaluate(scores): 
    pass #TODO


def main():
    # Load data 
    dirName = "data/archive/zero-indexed-files.txt"
    imgPath = "data/archive/Garage_classification/load/"

    df_raw = pd.read_csv(dirName,sep=' ')
   
    df_raw['image'] = imgPath + df_raw['image'].astype(str)
   

    print(df_raw.head()) #DEBUG

    #TODO:
    # convert df image column to opencv.imread
    # compute keypoints 
    # find optimal clusters K 
    # bin all data with clustering
    # cross-validated KNN 
    # classify using optimal K 
   
   #knn = KNeighborsClassifier()
    
    #pass #TODO 
    

if __name__ == '__main__':
    main()