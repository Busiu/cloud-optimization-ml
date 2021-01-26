import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

NUMBER_OF_CLUSTERS = 6
CLUSTER_SIZE_CUTOFF = 15  # approx 5% of each dataset

# Getting the data from the csv file, casting 'local'/'cloud' to 0/1
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

# We increase the number of initializations to get better results.
kmeans = KMeans(
    init="random",
    n_clusters=NUMBER_OF_CLUSTERS,
    n_init=100,
    max_iter=300
)

# Performing k-means on local data
print("Local # of records: ", len(data_local))
kmeans.fit(scaled_data_local)
print("Local SSE: ", kmeans.inertia_)

# Getting all clusters
clusters = []
for i in range(0, NUMBER_OF_CLUSTERS):
    cluster = scaled_data_local[kmeans.labels_ == i]
    clusters.append(cluster)

# For all points calculating distances from centers of their clusters
distances = []
for index, point in enumerate(scaled_data_local):
    center = kmeans.cluster_centers_[kmeans.labels_[index]]
    distance = np.linalg.norm(point - center)
    distances.append(distance)

# We discard all clusters with size less than an arbitrary cutoff
indices_to_discard_by_cluster_size = []
print("Cluster sizes: ")
for i, cluster in enumerate(clusters):
    size = len(cluster)
    print(size)
    if size < CLUSTER_SIZE_CUTOFF:
        cluster_indices = kmeans.labels_ == i
        indices_to_discard_by_cluster_size += list(np.array(range(len(data_local)))[cluster_indices])
print("Indices to discard by cluster size: ", indices_to_discard_by_cluster_size)

# We also discard all records with their in-cluster distances too high
sorted_distances = sorted(distances)
plt.style.use("fivethirtyeight")
plt.plot(range(len(sorted_distances)), sorted_distances)
plt.ylabel("In-cluster distances for local")
plt.show()

kl = KneeLocator(
    range(0, len(sorted_distances)), sorted_distances, curve="convex", direction="increasing"
)
print("Local in-cluster distances elbow point: ", kl.elbow)
number_of_bad_records = len(data_local) - kl.elbow - 1
print("Number of records to discard by high in-cluster distance: ", number_of_bad_records)
indices_to_discard_by_distance = list((-np.array(distances)).argsort()[:number_of_bad_records])
print("Indices to discard by in-cluster distance: ", indices_to_discard_by_distance)

all_indices_to_discard = list(set(indices_to_discard_by_cluster_size).union(set(indices_to_discard_by_distance)))
print("All indices to discard: ", all_indices_to_discard)

# Deleting all records by their indices (from highest to lowest, to retain proper indexing during the process).
data_local_refined = list(data_local)
for i in sorted(all_indices_to_discard, reverse=True):
    del data_local_refined[i]

# Doing all of the above again for cloud data
print("Cloud # of records: ", len(data_cloud))
kmeans.fit(scaled_data_cloud)
print("Cloud SSE: ", kmeans.inertia_)

clusters = []
for i in range(0, NUMBER_OF_CLUSTERS):
    cluster = scaled_data_cloud[kmeans.labels_ == i]
    clusters.append(cluster)

distances = []
for index, point in enumerate(scaled_data_cloud):
    center = kmeans.cluster_centers_[kmeans.labels_[index]]
    distance = np.linalg.norm(point - center)
    distances.append(distance)

indices_to_discard_by_cluster_size = []
print("Cluster sizes: ")
for i, cluster in enumerate(clusters):
    size = len(cluster)
    print(size)
    if size < CLUSTER_SIZE_CUTOFF:
        cluster_indices = kmeans.labels_ == i
        indices_to_discard_by_cluster_size += list(np.array(range(len(data_cloud)))[cluster_indices])
print("Indices to discard by cluster size: ", indices_to_discard_by_cluster_size)

sorted_distances = sorted(distances)
plt.style.use("fivethirtyeight")
plt.plot(range(len(sorted_distances)), sorted_distances)
plt.ylabel("In-cluster distances for cloud")
plt.show()

kl = KneeLocator(
    range(0, len(sorted_distances)), sorted_distances, curve="convex", direction="increasing"
)
print("Cloud in-cluster distances elbow point: ", kl.elbow)
number_of_bad_records = len(data_cloud) - kl.elbow - 1
print("Number of records to discard by high in-cluster distance: ", number_of_bad_records)
indices_to_discard_by_distance = list((-np.array(distances)).argsort()[:number_of_bad_records])
print("Indices to discard by in-cluster distance: ", indices_to_discard_by_distance)

all_indices_to_discard = list(set(indices_to_discard_by_cluster_size).union(set(indices_to_discard_by_distance)))
print("All indices to discard: ", all_indices_to_discard)

data_cloud_refined = list(data_cloud)
for i in sorted(all_indices_to_discard, reverse=True):
    del data_cloud_refined[i]

# Adding back missing columns
data_local_refined = list(map(lambda e: ('local',) + tuple(e) + (10,), data_local_refined))
data_cloud_refined = list(map(lambda e: ('cloud',) + tuple(e) + (10,), data_cloud_refined))
# Merging both datasets together and saving to a file
refined_data = data_local_refined + data_cloud_refined
print(np.array(refined_data))
np.savetxt("output.csv", refined_data, delimiter=", ", fmt='%s')
