import numpy as np
import pandas as pd

from numpy import genfromtxt
from sklearn import preprocessing

def preprocess(edit):
    edit.replace('?',np.NaN)
    edit = edit[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]]
    edit = edit.dropna()
    std_scale = preprocessing.StandardScaler().fit(edit)
    edit = std_scale.transform(edit)
    edit = preprocessing.normalize(edit)
    return edit

def MyDBSCAN(D, eps, MinPts):
    """
    Cluster the dataset `D` using the DBSCAN algorithm.
    
    MyDBSCAN takes a dataset `D` (a list of vectors), a threshold distance
    `eps`, and a required number of points `MinPts`.
    
    It will return a list of cluster labels. The label -1 means noise, and then
    the clusters are numbered starting from 1.
    """
 
    # This list will hold the final cluster assignment for each point in D.
    # There are two reserved values:
    #    -1 - Indicates a noise point
    #     0 - Means the point hasn't been considered yet.
    # Initially all labels are 0.    
    labels = [0]*len(D)

    # C is the ID of the current cluster.    
    C = 0
    
    # This outer loop is just responsible for picking new seed points--a point
    # from which to grow a new cluster.
    # Once a valid seed point is found, a new cluster is created, and the 
    # cluster growth is all handled by the 'expandCluster' routine.
    
    # For each point P in the Dataset D...
    # ('P' is the index of the datapoint, rather than the datapoint itself.)
    for P in range(0, len(D)):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        NeighborPts = regionQuery(D, P, eps)
        
        # If the number is below MinPts, this point is noise. 
        # This is the only condition under which a point is labeled 
        # NOISE--when it's not a valid seed point. A NOISE point may later 
        # be picked up by another cluster as a boundary point (this is the only
        # condition under which a cluster label can change--from NOISE to 
        # something else).
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    
    # All data has been clustered!
    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Grow a new cluster with label `C` from the seed point `P`.
    
    This function searches through the dataset to find all points that belong
    to this new cluster. When this function returns, cluster `C` is complete.
    
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.  
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    # Look at each neighbor of P (neighbors are referred to as Pn). 
    # NeighborPts will be used as a FIFO queue of points to search--that is, it
    # will grow as we discover new branch points for the cluster. The FIFO
    # behavior is accomplished by using a while-loop rather than a for-loop.
    # In NeighborPts, the points are represented by their index in the original
    # dataset.
    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # If Pn was labelled NOISE during the seed search, then we
        # know it's not a branch point (it doesn't have enough neighbors), so
        # make it a leaf point of cluster C and move on.
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighborPts = regionQuery(D, Pn, eps)
            
            # If Pn has at least MinPts neighbors, it's a branch point!
            # Add all of its neighbors to the FIFO queue to be searched. 
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
            # If Pn *doesn't* have enough neighbors, then it's a leaf point.
            # Don't queue up it's neighbors as expansion points.
            #else:
                # Do nothing                
                #NeighborPts = NeighborPts               
        
        # Advance to the next point in the FIFO queue.
        i += 1        
    
    # We've finished growing cluster C!


def regionQuery(D, P, eps):
    """
    Find all points in dataset `D` within distance `eps` of point `P`.
    
    This function calculates the distance between a point P and every other 
    point in the dataset, and then returns only those points which are within a
    threshold distance `eps`.
    """
    neighbors = []
    
    # For each point in the dataset...
    for Pn in range(0, len(D)):
        
        # If the distance is below the threshold, add it to the neighbors list.
        if np.linalg.norm(D[P] - D[Pn]) < eps:
           neighbors.append(Pn)
            
    return neighbors

def visualizeCluster(labeled_data):
    cluster = dict()
    i = 1
    for data in labeled_data:
        if data in cluster.keys():
            cluster[data].append(i)
        else:
            cluster[data] = [i]
        i += 1

    print('\n')
    n_unclusterd_data = 0
    if -1 in cluster:
        unclustered = cluster.pop(-1)
        n_unclusterd_data = len(unclustered)
        print("Data noise :")
        print(unclustered)

    keys = sorted(list(cluster.keys()))
    print('\n')
    for key in keys:
        print("Data dalam cluster " + str(key) + " :")
        print(cluster[key])
        print("/// cluster " + str(key) + "\n")

    n_cluster = len(keys)
    print('\n')
    print("Terdapat " + str(n_cluster) + " cluster")
    print("Jumlah data yang tidak memiliki cluster (noise) : " + str(n_unclusterd_data))
    for key in keys:
        print("Cluster " + str(key) + " memiliki data sebanyak: " + str(len(cluster[key])))

def calculateAccuracy(predicted_labels, correct_labels):

    correct_labels1 = []
    correct_labels2 = []
    for label in correct_labels:
        if label == 0:
            correct_labels1.append(1)
            correct_labels2.append(2)
        else:
            correct_labels1.append(2)
            correct_labels2.append(1)

    correct1 = 0
    correct2 = 0

    if max(predicted_labels) == 2:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == correct_labels1[i]:
                correct1 += 1
            if predicted_labels[i] == correct_labels2[i]:
                correct2 += 1

    print("Predicted labels : " + str(len(predicted_labels)))
    print("Correct labels1 : " + str(correct1))
    print("Correct labels2 : " + str(correct2))
    accuracy1 = correct1 / len(predicted_labels)
    accuracy2 = correct2 / len(predicted_labels)
    return max(accuracy1, accuracy2)

if __name__ == "__main__":
    data = pd.read_csv('censusincome.csv')

    labels = data["income"]
    le = preprocessing.LabelEncoder().fit(labels)
    labels = le.transform(labels)

    new_data = preprocess(data)
    print(new_data)

    ### Ini training semua tapi lama banget kayaknya :v
    ### Ini epsilonnya 1 hasil kira2
    # print(MyDBSCAN(new_data,1,1000))

    ### Coba training kalo datanya cuma 10
    cut_data = new_data[0:3000]

    predicted_labels = MyDBSCAN(cut_data,0.8,450)
    visualizeCluster(predicted_labels)
    accuracy = calculateAccuracy(predicted_labels, labels)
    print("Akurasi : " + str(accuracy * 100) + "%")

    ### Coba ngecek jarak -> buat ngira2 epsilon
    # for Pn in range(0, len(new_data)) :
    # 	print(np.linalg.norm(new_data[0] - new_data[Pn]))
    
