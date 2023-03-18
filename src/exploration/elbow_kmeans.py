# elbow_kmeans.py
# Elbow method for optimizing the number of clusters in K-Means. Using inertia (euclidean distance)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np 



def elbow_kmeans(data): 
    # Optimize
    # Euclidean distance between clusters, 'Inertia'
    dist = []
    K = []

    for k in range(1,60,5): #Minimum is number of classes
        kmeans = KMeans(n_clusters=k,random_state=0)
        kmeans.fit(data)
        dist.append(kmeans.inertia_)
        K.append(k)
    
    plt.plot(dist)
    plt.xlabel("K")
    plt.ylabel("Distance")
    plt.show()

