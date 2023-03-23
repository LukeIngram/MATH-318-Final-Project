# elbow_kmeans.py
# Elbow method for optimizing the number of clusters in K-Means. Using inertia (euclidean distance)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 
from tqdm import tqdm 



def elbow_kmeans(data,kmax=60): 
    # Optimize
    # Euclidean distance between clusters, 'Inertia'
    dist = []
    K = []

    for k in tqdm(range(1,kmax,3)): #Minimum is number of classes
        kmeans = KMeans(n_clusters=k,random_state=0)
        kmeans.fit(data)
        dist.append(kmeans.inertia_)
        K.append(k)
    
    plt.plot(K,dist,'bx-')
    plt.xlabel("K")
    plt.ylabel("Distance")
    plt.savefig("elbow_kmeans.png")
    plt.show()
