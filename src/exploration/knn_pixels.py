# knn_pixels.py
# monolithic script to evaluate knn classification with raw pixel data
# NOTE: DO NOT RUN OUTSIDE OF HEAVY COMPUTE ENVIRONMENT

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
from 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from tqdm import tqdm

import cv2
import glob


# K-fold Cross-validated KNN using sklearn. & plots score h
def crossValidate(X,Y,folds=10,kmax = 10):
    kscores = []
    for i in tqdm(range(1,kmax)):
        knn = KNeighborsClassifier(n_neighbors=i,n_jobs=8) # 5 parallel tasks to speed things up
        cv = cross_val_score(knn,X,Y,cv=folds,scoring="accuracy")
        kscores.append(cv.mean())
    
    plt.plot(list(range(1,kmax)),kscores)  
    plt.savefig("Optimal_neighbors_pixel.png")
    plt.show()



def main():
 
    # Load & flatten entire dataset to memory
    imgPaths = glob.glob("../src/data/archive/Garbage_classification/load/*.jpg")
    X_raw = np.array([np.array(cv2.imread(img)) for img in imgPaths])
    X_data = X_raw.flatten().reshape(2527,589824) # flatten & reshape to retain features

    df = pd.read_csv("data/archive/zero-indexed-files.txt",sep=' ')
   
    df['image'] = [i for i in  X_data]

    print(df.head()) #DEBUG
    
    X_train,X_test,Y_train,Y_train = train_test_split(df['image'],df['class'],
                                                     test_size=0.20,random_state=42,stratify=df['class'])
    
    # Reduce down to 95% explained variance
    scaler = MinMaxScaler()
    scaler.fit(X_train) 

    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_train)

    pca = PCA(n_components = 0.95) # n_components < 1 converts to pc's needed for that expl. var.
    pca.fit(Xs_train) 
    Xs_train_reduced = pca.transform(Xs_train) #<------ uncommnet to run 
    Xs_test_reduced = pca.transform(Xs_test)

    print(f"Dimensions of data after PCA: {Xs_train_reduced.shape}") 

    # find optimal neighbors
    crossValidate(Xs_train_reduced,Y_train,)

    knn = KNeighborsClassifier()

