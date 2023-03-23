# knn_SIFT.py 
# Monolithic script to evaluate the KNN classifier with descriptors generated from SIFT 

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
    # normalize
    norm = cv2.normalize(img,np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX) # CAUSE OF ERROR?
    sift = cv2.SIFT_create() 
    kps,des = sift.detectAndCompute(norm,None) #FIX: OUTPUTS NULLS
    if (len(kps) < 1): 
        print("NULL HERE")

    return kps,des
    


# Using K-Means clustering for feature reduction. 
# Optimal K is determined by elbow method (see elbow_kmeans.py)
def cluster(descriptors,k = 15): #TODO 
    clusters = KMeans(k,random_state=42).fit(descriptors)
    return clusters 

# Data binning through normalized histograms. 
def binData(keypoints,descriptors,clusters):
    hists = []
    for kps,des in zip(keypoints,descriptors):
        hist = np.zeros(len(clusters.labels_))
        normFact = np.size(kps)
        bin = clusters.predict([des])
        hist[bin] += 1/normFact
        hists.append(hist)
    return hists

# K-fold Cross-validated KNN using sklearn. & plots score h
def crossValidate(X,Y,folds=10,kmax = 10):
    kscores = []
    for i in tqdm(range(1,kmax)):
        knn = KNeighborsClassifier(n_neighbors=i,n_jobs=8) # 5 parallel tasks to speed things up
        cv = cross_val_score(knn,X,Y,cv=folds,scoring="accuracy")
        kscores.append(cv.mean())
    
    plt.plot(list(range(1,kmax)),kscores)  
    plt.savefig("Optimal_neighbors_sift.png")
    plt.show()

    

# Evaluate classifier & display confusion matrix
def evaluate(scores): 
    pass #TODO


def main():

    # Preproccessing

    # Load data 
    dirName = "data/archive/zero-indexed-files.txt"
    imgPath = "data/archive/Garbage_classification/load/"

    df = pd.read_csv(dirName,sep=' ')
   
    df['image'] = imgPath + df['image'].astype(str)
    df['image'] = df['image'].apply(lambda x: cv2.imread(x))

    print(df.head()) #DEBUG
    
    train_X,test_X,train_Y,test_Y = train_test_split(df['image'],df['class'],
                                                     test_size=0.33,random_state=42,stratify=df['class'])


    # Fetch keypoints from training data
    train_keys = []
    train_des = []
    for sample in train_X: 
        kps,des = SIFT(sample)
        train_keys.append(kps)
        for d in des: 
            train_des.append(d)

    # find optimal clustering
    #elbow_kmeans(train_des,kmax=60)

    # cluster data with said optimal value (from elbow)
    kmeans = cluster(train_des,k = 60)

    # Histogram with new clusters
    train_hists = binData(train_keys,train_des,kmeans)

    #Now Histogram the testing data using kmeans from training
    test_keys = []
    test_des = []
    for sample in test_X: 
        kps,des = SIFT(sample)
        test_keys.append(kps)
        for d in des: 
            test_des.append(d)

    test_hists = binData(test_keys,test_des,kmeans)


    print("CV")
    crossValidate(train_hists,train_Y,kmax=50)

    # Fit Optimal
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(train_hists,train_Y)

    res = knn.predict(test_hists)

    print(classification_report(test_Y,res,target_names=["Glass","Paper","Cardboard","Plastic","Metal","Trash"]))

    #TODO:
    # bin all data with clustering
    # cross-validated KNN 
    # classify using optimal K 
   
   #knn = KNeighborsClassifier()
    
    #pass #TODO
    

if __name__ == '__main__':
    main()