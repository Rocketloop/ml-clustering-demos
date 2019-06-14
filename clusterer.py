"""Docstring: Demo of clustering algorithms in python.

This programm uses sklearn and hdbscan libraries to train clustering
algorithms in python.
"""

__author__ = "Tobias Marzell"
__credits__ = ""
__email__ = "tobias.marzell@gmail.com"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import hdbscan

def remove_incomplete_data(data):
	""" Remove every row of the data that contains atleast 1 "?". """
	return data.replace("?", np.nan).dropna(0, "any")

def get_dummies(data, categorical_features):
	""" Get the dummies of the categorical features for the given data. """
	for feature in categorical_features:
		# create dummyvariable with pd.get_dummies and drop all categorical variables with dataframe.drop
		data = data.join(pd.get_dummies(data[feature], prefix=feature, drop_first=True)).drop(feature, axis=1)
	return data

def plot_clustering(data, k=None, vars=None):
	""" Plot the clustered data. """
	if vars == None:
		cols = list(data.columns)
	else:
		vars.append('cluster')
		vars = set(vars)
		vars = list(vars)
		cols = vars
	g = sns.pairplot(data[cols], hue='cluster', diag_kind='hist')
	if not k == None:
		plt.subplots_adjust(top=0.9)
		g.fig.suptitle(k + 2)
		g.fig.tight_layout()
	plt.subplots_adjust(left=0.05, bottom=0.05)
	plt.show()

def get_kmeans(data, n_clusters=3):
	""" Do kmeans clustering and return clustered data """
	kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300)
	vals = data.iloc[ :, 0:].values
	y_pred = kmeans.fit_predict(StandardScaler().fit_transform(vals))
	data["cluster"] = y_pred
	return data, kmeans.inertia_

def get_optics(data):
	""" Do optics clustering and return clustered data """
	optics = OPTICS(min_samples=50)
	vals = data.iloc[ :, 0:].values
	y_pred = optics.fit_predict(StandardScaler().fit_transform(vals))
	data["cluster"] = y_pred
	return data, labels

def get_dbscan(data, min_samples=3):
	""" Do dbscan clustering and return clustered data """
	db = DBSCAN(eps=0.37, min_samples=min_samples)
	vals = data.iloc[ :, 0:].values
	y_pred = db.fit_predict(StandardScaler().fit_transform(vals))
	data["cluster"] = y_pred
	return data

def get_hdbscan(data, min_cluster_size=4):
	hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
	vals = data.iloc[ : ,0: ].values
	y_pred = hdb.fit_predict(StandardScaler().fit_transform(vals))
	data["cluster"] = y_pred
	return data, hdb

def main():
	""" Run the script with given datasets and settings. """
	data = pd.read_csv("./datasets/points.txt")
	data2 = data
	missing = data.isna().sum()
	data = remove_incomplete_data(data)
	# delete all rows with missing data
	#data = data.drop(["Country"], axis = 0)
	#data = data.dropna(0, "any")
	#cat_vars = ["Region"]

	#	othervars = list(set(list(data.columns.values))-set(cat_vars))
	#data_with_dummies = get_dummies(data, cat_vars)
	#data = data_with_dummies
	kmeans = []
	elbow = []
	calinskis = []
	silhouettes = []
	number_clusters = []
	#plt.plot(data['y'], data['x'], 'o')
	#plt.title("Data")
	#plt.show()
	for i in range(10):
		temp1, temp2 = get_kmeans(data, i+2)
		kmeans.append(data.merge(temp1['cluster']))
		elbow.append(temp2)
		print(metrics.calinski_harabasz_score(data2, data['cluster']), metrics.silhouette_score(data2, data['cluster']), (i+2))
		calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
		silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
		number_clusters.append(i+2)
	plt.plot(elbow,'ro-', label="Elbow")
	plt.title("KMeans Elbow")
	plt.show()
	plt.plot(number_clusters, calinskis, 'ro-', label="KMeans Ralinski Harabasz Score")
	plt.title("KMeans Calinski Harabasz Score")
	plt.xlabel("number of clusters")
	plt.show()
	plt.plot(number_clusters, silhouettes,'ro-', label="KMeans Silhouette Score")
	plt.title("KMeans Silhouette Score")
	plt.xlabel("number of clusters")
	plt.show()
	plt.show()
	kmean,temp = get_kmeans(data)
	kmean = data.merge(kmean['cluster'])
	#print(metrics.calinski_harabasz_score(data2, data['cluster']), metrics.silhouette_score(data2, data['cluster']))
	#plot_clustering(kmeans[i])
	#kmean = data.merge(get_kmeans(data)['cluster'])
	#print(metrics.calinski_harabasz_score(data2, data['cluster']), metrics.silhouette_score(data2, data['cluster']))
	plot_clustering(kmean)
	calinskis = []
	silhouettes = []
	min_samples = []
	for i in range (8):
		dbsca = data.merge(get_dbscan(data, i+3)['cluster'])
		calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
		silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
		print(metrics.calinski_harabasz_score(data2, data['cluster']), metrics.silhouette_score(data2, data['cluster']))
		min_samples.append((i+3))
	plt.plot(min_samples, calinskis, 'ro-', label="DBSCAN Ralinski Harabasz Score")
	plt.title("DBSCAN Calinski Harabasz Score")
	plt.xlabel("minimum samples")
	plt.show()
	plt.plot(min_samples, silhouettes,'ro-', label="DBSCAN Silhouette Score")
	plt.title("DBSCAN Silhouette Score")
	plt.xlabel("minimum samples")
	plt.show()
	dbsca = data.merge(get_dbscan(data)['cluster'])
	plot_clustering(dbsca)
	calinskis = []
	silhouettes = []
	min_cluster_size = []
	for i in range (8):
            hdbsca, clusterer = get_hdbscan(data, i+3)
            hdbsca = data.merge(hdbsca)
            calinskis.append(metrics.calinski_harabasz_score(data2, data['cluster']))
            silhouettes.append(metrics.silhouette_score(data2, data['cluster']))
            min_cluster_size.append(i+3)
            print(metrics.calinski_harabasz_score(data2, data['cluster']), metrics.silhouette_score(data2, data['cluster']))
	plt.plot(min_cluster_size, calinskis,'ro-', label="HDBSCAN Ralinski Harabasz Score")
	plt.title("HDBSCAN Calinski Harabasz Score")
	plt.xlabel("minimum cluster size")
	plt.show()
	plt.plot(min_cluster_size, silhouettes,'ro-', label="HDBSCAN Silhouette Score")
	plt.title("HDBSCAN Silhouette Score")
	plt.xlabel("minimum cluster size")
	plt.show()
	hdbsca, clusterer = get_hdbscan(data)
	hdbsca = data.merge(hdbsca['cluster'])
	clusterer.minimum_spanning_tree_.plot(edge_cmap="rainbow", edge_alpha=0.6, node_size=80, edge_linewidth=2)
	plt.show()
	clusterer.single_linkage_tree_.plot(cmap="rainbow", colorbar=True)
	plt.show()
	clusterer.condensed_tree_.plot(cmap="rainbow")
	plt.show()
	plot_clustering(hdbsca)
if __name__ == "__main__":
	main()
