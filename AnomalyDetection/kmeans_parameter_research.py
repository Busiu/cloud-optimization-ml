import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Getting the data from the csv file, casting local/cloud to 0/1
datafile = "xd.csv"
data = np.genfromtxt(
    datafile,
    delimiter=",",
    skip_header=1,
    converters={0: lambda s: 0 if s == b'local' else 1}
)

# Converting the data to np 2d array
data2d = np.array(list(map(list, data)))

# Splitting the data into local/cloud, dropping the last column (constant)
data_local = data2d[data2d[:, 0] == 0, 1:5]
data_cloud = data2d[data2d[:, 0] == 1, 1:5]

# Scaling the data to mean=0, std=1 to enable k-means
scaler = StandardScaler()
scaled_data_local = scaler.fit_transform(data_local)
scaled_data_cloud = scaler.fit_transform(data_cloud)

# Following code allows to choose the best number of clusters through the elbow method
# and silhouette coefficient. Due to nondeterministic character of the algorithm
# it is advised to repeat this code a couple of times for consistent conclusions.
kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300
}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data_local)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
print("Local data elbow point: ", kl.elbow)

silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data_local)
    score = silhouette_score(scaled_data_local, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# The same as above, but for cloud data.
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data_local)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(
    range(1, 11), sse, curve="convex", direction="decreasing"
)
print("Cloud data elbow point: ", kl.elbow)

silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_data_local)
    score = silhouette_score(scaled_data_local, kmeans.labels_)
    silhouette_coefficients.append(score)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), silhouette_coefficients)
plt.xticks(range(2, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()

# For both datasets, the silhouette coefficient tends to drop heavily at k=4
# (rarely at k=5 or k=6), the elbow point tends to be at k=3 (rarely at k=4 or k=5),
# so it seems wise to choose k=3. However, as we use the clustering for anomaly
# detection, it is convenient to have k a little bit higher, so we get outliers
# in their own small clusters. Therefore, k=6 was chosen.
