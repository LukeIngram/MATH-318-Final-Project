# elbow_kmeans.py
# Elbow method for optimizing the number of clusters in K-Means. Using inertia (euclidean distance)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 



def elbow_kmeans(data): 
    # Optimize
    # Euclidean distance between clusters, 'Inertia'
    dist = []
    K = []

    #data_long = data.ravel().reshape(-1,1)

    for k in tqdm(range(1,60,5)): #Minimum is number of classes
        kmeans = KMeans(n_clusters=k,random_state=0)
        kmeans.fit(data)
        dist.append(kmeans.inertia_)
        K.append(k)
    
    plt.plot(dist)
    plt.xlabel("K")
    plt.ylabel("Distance")
    plt.show()

