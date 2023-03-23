# knn_SIFT.py 
# Monolithic script to evaluate the KNN classifier with descriptors generated from SIFT 

from sklearn.model_selection import train_test_split,cross_val_score
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
    normalized = cv2.normalize(img,np.zeros(img.shape),0,255,cv2.NORM_MINMAX) # normalized image helps reduce keypoint nulls
    sift = cv2.SIFT_create() 
    kps,des = sift.detectAndCompute(normalized,None) #FIX: OUTPUTS NULLS
    if (len(kps) < 1): 
        print("NULL HERE")

    return kps,des
    


# Using K-Means clustering for feature reduction. 
# Optimal K is determined by elbow method (see elbow_kmeans.py)
def cluster(descriptors,k = 60): #TODO 
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
        knn = KNeighborsClassifier(n_neighbors=i,n_jobs=5) # 5 parallel tasks to speed things up
        cv = cross_val_score(knn,X,Y,cv=folds,scoring="accuracy")
        kscores.append(cv.mean())
    
    plt.plot(list(range(1,kmax)),kscores)  
    plt.show()



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

    # Fetch keypoints from training data
    train_keys = []
    train_des = []
    for sample in train_X: 
        kps,des = SIFT(sample)
        train_keys.append(kps)
        for d in des: 
            train_des.append(d)

    
    # find optimal clustering
    # elbow_kmeans(np.array(train_des).ravel())

    # cluster data with said optimal value (from elbow)
    kmeans = cluster(train_des,k = 15)

    # Histogram with new clusters
    train_hists = binData(train_keys,train_des,kmeans)

    print(train_X.shape) #DEBBUG
    print(len(train_hists)) #DEBUG

    #Now Histogram the testing data using kmeans from training
    test_keys = []
    test_des = []
    for sample in test_X: 
        kps,des = SIFT(sample)
        test_keys.append(kps)
        for d in des: 
            test_des.append(d)


    test_hists = binData(test_keys,test_des,kmeans)

    print(test_X.shape) #DEBUG
    print(len(test_hists)) #DEBUG
    
    crossValidate(train_hists,train_Y)

    # Fit Optimal
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.array(train_hists),train_Y)

    res = knn.predict(np.array(test_hists))

    print(classification_report(test_Y,res,target_names=["Glass","Paper","Cardboard","Plastic","Metal","Trash"]))

    #TODO:
    # bin all data with clustering
    # cross-validated KNN 
    # classify using optimal K 
   
   #knn = KNeighborsClassifier()
    
    #pass #TODO
    

if __name__ == '__main__':
    main()