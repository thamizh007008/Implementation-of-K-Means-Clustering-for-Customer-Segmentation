# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required Python libraries and load the Mail_Customers.csv dataset.
2.Select the relevant numerical features and preprocess the data if required.
3.Apply the K-Means Clustering algorithm by choosing the number of clusters and fit the model to the customer data.
4.Visualize the clusters using a scatter plot and analyze the customer segments formed.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MAALINI B N
RegisterNumber:  212224060136
*/
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
X = data.iloc[:, [3, 4]].values
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
plt.figure(figsize=(8, 6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], label='Cluster 5')
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    label='Centroids'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-Means Clustering for Customer Segmentation")
plt.legend()
plt.show()
```

## Output:
<img width="710" height="135" alt="image" src="https://github.com/user-attachments/assets/1d31b594-5e48-4475-a60e-20753f7244b4" />
<img width="291" height="140" alt="image" src="https://github.com/user-attachments/assets/5a47a71e-f0e0-48af-9939-4b1ef26d7cb1" />
<img width="695" height="545" alt="image" src="https://github.com/user-attachments/assets/9e017099-811a-4666-8715-11c5ccd894d2" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
