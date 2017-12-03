import numpy as np
import pandas as pd

from numpy import genfromtxt
from sklearn import preprocessing

def preprocess(edit):
    """
    Melakukan preproses terhadap data, data yang digunakan hanya data numerik
    """
    edit.replace('?',np.NaN)
    edit = edit[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]]
    edit = edit.dropna()
    std_scale = preprocessing.StandardScaler().fit(edit)
    edit = std_scale.transform(edit)
    edit = preprocessing.normalize(edit)
    return edit

def MyDBSCAN(D, eps, MinPts):
    """
    Parameter :
      D         : Dataset
      eps       : Epsilon (jarak range maksimal)
      MinPts    : Jumlah point yang dibutuhkan untuk membentuk suatu cluster
   
      Melakukan cluster dengan algorima DBScan.
      Kluster dimulai dari angka 1 dan seterusnya

    """
 
    # labels berisi status dari point (data)
    #    -1 : noise
    #     0 : belum ditentukan
    # Semua label diinisialiasi dengan 0
    labels = [0]*len(D)

    # C adalah kluater saat ini   
    C = 0
    
    # Mencari seed point
    for P in range(0, len(D)):
    
        
        if not (labels[P] == 0):
           continue
        
        # Cari neigbor dari point P (pint yang jaraknya < eps)
        NeighborPts = regionQuery(D, P, eps)
        
        
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        
        else: 
           C += 1
           growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    
    
    return labels


def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    
    Parameters:
      D             : Dataset
      labels        : Array label untuk semua dataset
      P             : Indeks dari seed point (data)
      NeighborPts   : Tetangga dari P (yang berjarak < eps)
      C             : Label dari cluster
      eps           : Epsilon (jarak range maksimal)
      MinPts - Minimum required number of neighbors

    """

    labels[P] = C
        
    i = 0
    while i < len(NeighborPts):    
        
        Pn = NeighborPts[i]
        

        if labels[Pn] == -1:

           # Pn bukan branch point    
           labels[Pn] = C
        
        elif labels[Pn] == 0:
           
            labels[Pn] = C
            
            # Cari semua neighbor Pn
            PnNeighborPts = regionQuery(D, Pn, eps)
            
            # Jika neighbor Pn >= MinPts maka Pn branch point
            if len(PnNeighborPts) >= MinPts:

                # Neighbour Pn ditambahkan ke daftar unutk perncarian selanjutnya
                NeighborPts = NeighborPts + PnNeighborPts
            
        i += 1        

def regionQuery(D, P, eps):

    """
    
    Parameter :
      D   : Dataset
      P   : Point (data yangs sedang ditinjau)
      eps : Epsilon (jarak range maksimal)
    
    regiOn query digunakan untuk menemukan semua point (data) pada dataset D
    yang berjarak < eps dari point P (data ynags edang ditinjau)
    
    """
    neighbors = []
    
    # Iterasi setiap poin pada dataset
    for Pn in range(0, len(D)):
        
        # Jika jarak sebuah poin ke point P < eps, tambahkan pada neighbours
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
    label1 = 0
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

    # Load data
    # data = pd.read_csv('censusincome.csv')
    data = pd.read_csv('censustest.csv')

    # Melakukan preprocess data
    labels = data["income"]
    le = preprocessing.LabelEncoder().fit(labels)
    labels = le.transform(labels)

    new_data = preprocess(data)
    # print(new_data)

    # Data di trim karena terlalu besar (memori tidak kuat)
    cut_data = new_data[0:30000]

    # Cari cluter dengan DBScan dan menampilkan hasil serta akurasinya
    predicted_labels = MyDBSCAN(cut_data,0.77,450)
    visualizeCluster(predicted_labels)
    accuracy = calculateAccuracy(predicted_labels, labels)
    print("Akurasi : " + str(accuracy * 100) + "%")

    
    
