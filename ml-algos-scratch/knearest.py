import numpy as np
import pandas as pd
import heapq

df = pd.read_csv('datasets/Iris.csv', index_col=0)

features = df.drop(columns='Species').to_numpy()
labels = df['Species']

mean = np.mean(features,axis=0)
std = np.std(features, axis=0)

scaled_features = (features-mean)/std

def distance(data):
    scaled_data = (data-mean)/std
    distances = np.sum((scaled_features - scaled_data)**2, axis=1)
    return [(dist, index) for index, dist in enumerate(distances)]

def knn(k, data):
    indices = np.array([idx[-1] for idx in heapq.nsmallest(k,distance(data), key=lambda x: x[0])])
    label = df.loc[indices, 'Species'].value_counts().idxmax()
    return label

print(knn(5, np.array([1,2,3,4])))