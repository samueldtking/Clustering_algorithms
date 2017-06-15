## KMEANS TEMPLATE (Python)


# Import modules 
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#### USER INPUTS ####
seed = random.randint(0,1000)
sample_size = 500
number_of_clusters = 2
number_of_runs = 10

## PARAMETER TUNNING ###
algorithm='auto' 
max_iter=300
tol=0.0001
precompute_distances='auto' 
n_jobs=1
#document:   http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


##Base code##

# Set seeds 
np.random.seed(seed)

# load toy data (Iris flower numpy array from sklearn library in this case)
digits = load_iris()
data = scale(digits.data)

# convert numpy to panda (as more common input use - all variables floats)
df = pd.DataFrame(data)
print(df.shape)
data = df

# create labels 
n_samples, n_features = data.shape
n_clusters = number_of_clusters
labels = digits.target

# print shape of dataset 
print("n_clusters: %d, \t n_samples %d, \t n_features %d"
      % (n_clusters, n_samples, n_features))


print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

# define UDF for kmeans 
def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

# initialize K-means algorithm with centroids placed using K-mean++
bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, 
                     n_init=number_of_runs, 
                     algorithm=algorithm, 
                     max_iter=max_iter, 
                     tol=tol, 
                     precompute_distances=precompute_distances, 
                     n_jobs=n_jobs),
            name="k-means++", data=data)





# in the case seed is set to a deterministic number, n_init need only be run one as there will be no change
pca = PCA(n_components=n_clusters).fit(data)
bench_k_means(KMeans(init='k-means++', n_clusters=n_clusters, 
                     n_init=number_of_runs, 
                     algorithm=algorithm, 
                     max_iter=max_iter, 
                     tol=tol, 
                     precompute_distances=precompute_distances, 
                     n_jobs=n_jobs),
              name="PCA-based",
              data=data)
print(79 * '_')




##Visualizing the results on a PCA-reduced surface##

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=number_of_runs)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2,color='k')
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=number_of_clusters)
plt.title('K-means clustering on the Iris dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

