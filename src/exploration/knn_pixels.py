# knn_pixels.py
# monolithic script to evaluate knn classification with raw pixel data
# NOTE: DO NOT RUN OUTSIDE OF HEAVY COMPUTE ENVIRONMENT

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import glob



def main():
    # Load & flatten entire dataset to memory
    imgPaths = glob.glob("../src/data/archive/Garbage_classification/load/*.jpg")
    X_raw = np.array([np.array(cv2.imread(img)) for img in imgPaths])
    X_data = X_raw.flatten().reshape(2527,589824) # flatten & reshape to retain features

    X_train,X_test,Y_train,Y_test = train_test_split(X_data,)

    # Reduce down to 95% explained variance
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data) 
    pca = PCA(n_components = 0.95) # n_components < 1 converts to pc's needed for that expl. var.
    pca.fit(X_scaled) 
    X_reduced = pca.transform(X_scaled) #<------ uncommnet to run 

    print(f"Dimensions of data after PCA: {X_reduced.shape}") 

def classify(): #TODO
    pass


def evaluate(): #TODO
    pass
