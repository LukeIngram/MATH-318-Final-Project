# knn_SIFT.py 
# Monolithic script to evaluate the KNN classifier with descriptors generated from SIFT 

from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from elbow_kmeans import elbow_kmeans
from tqdm import tqdm 
import numpy as np 
import pandas as pd
import cv2
import glob


# SIFT obtains & returns image descriptors
def SIFT(img): 
    sift = cv2.SIFT_create() 
    kps,des = sift.detectAndCompute(img,None)
    
    return kps if kps is not None else [],des if des is not None else []
    

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
    imgPath = "data/archive/Garbage_classification/load/"

    df = pd.read_csv(dirName,sep=' ')
   
    df['image'] = imgPath + df['image'].astype(str)
    df['image'] = df['image'].apply(lambda x: cv2.imread(x))

    print(df.head()) #DEBUG
    
    train_X,test_X,train_Y,test_Y = train_test_split(df['image'],df['class'],
                                                     test_size=0.33,random_state=0)

    print(df.tail(1))

    train_des= []

    print(train_X[:-1])

    # load & cluster
    for sample in train_X: 
        kps,des = SIFT(sample)
        #print(des)
        for d in des:
            train_des.append(d)

    # find optimal clustering
    elbow_kmeans(train_des)

    # Bin data with clustering 
    
    

    #TODO:
    # find optimal clusters K 
    # bin all data with clustering
    # cross-validated KNN 
    # classify using optimal K 
   
   #knn = KNeighborsClassifier()
    
    #pass #TODO
    

if __name__ == '__main__':
    main()