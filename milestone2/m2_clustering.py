"""
CSE 517A Machine Learning
Application Project, Milestone 2

[optional - Clustering] 
    Cluster the features of your dataset (not using the labels). 
    Evaluate the clusters using the observed data/labels.
    
# --- #    
# Materials:
- 2.3. Clustering
    > http://scikit-learn.org/stable/modules/clustering.html
"""
#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc, setUsedData
# Read data
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)

trainBatchSize = 1000
testBatchSize = 300

#%% 2.1 K-Means
# sklearn.cluster.KMeans
# > http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# K-means Clustering
# > http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html
import numpy as np
from sklearn.cluster import KMeans

estimators = [('k_means_iris_7', KMeans(n_clusters=7))]
              #('k_means_iris_3', KMeans(n_clusters=3)),
              #('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
              #                                init='random'))]
fignum = 1
titles = ['7 clusters']
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('batch',trainX, trainY, testX, testY, trainBatchSize, testBatchSize)
for name, est in estimators:
    est.fit(usedTrainX)
    labels = est.labels_

# acc = np.sum(labels==usedTrainY)/trainBatchSize
# Order is different!

#%% 2.2 K_Means Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Compress 4 binary area features into single 'discrete' feature for visualization
def compressWildArea(usedTrainX):
    usedTrainX['Wilderness_Area'] = usedTrainX['Wilderness_Area1']*0 +\
                                    usedTrainX['Wilderness_Area2']*1 +\
                                    usedTrainX['Wilderness_Area3']*2 +\
                                    usedTrainX['Wilderness_Area4']*3
    return usedTrainX

# Compress 40 binary soil type features into single 'discrete' feature for visualization
def compressSoilType(usedTrainX):
    soilLabels = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
    usedTrainX['Soil_Type'] = (usedTrainX[soilLabels]*np.arange(0,40)).sum(1)
    return usedTrainX

# 3D plot    
def plotResult(usedTrainX, featureList, labels, title):
    if 'Wilderness_Area' in featureList:
        usedTrainX = compressWildArea(usedTrainX.copy())    
    if 'Soil_Type' in featureList:
        usedTrainX = compressSoilType(usedTrainX.copy())
    
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(usedTrainX[featureList[0]], usedTrainX[featureList[1]], usedTrainX[featureList[2]],
               c=labels.astype(np.float))#, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel(featureList[0])
    ax.set_ylabel(featureList[1])
    ax.set_zlabel(featureList[2])
    ax.set_title(title)
    ax.dist = 12
    
# Fell free to use any 3 features    
featureList = ['Elevation','Hillshade_9am','Wilderness_Area']
#featureList = ['Elevation','Horizontal_Distance_To_Roadways','Wilderness_Area']
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('all',trainX, trainY, testX, testY)
estimators = [('k_means_iris_7', KMeans(n_clusters=7))]
              #('k_means_iris_3', KMeans(n_clusters=3)),
              #('k_means_iris_bad_init', KMeans(n_clusters=3, n_init=1,
               #                                init='random'))]

figIdx = 0
titles = ['7 clusters']
for name, est in estimators:    
    est.fit(usedTrainX)
    plotResult(usedTrainX, featureList, est.labels_, titles[figIdx], wildMode = True)
    figIdx = figIdx + 1
    
# True data 3D Plot
true_labels = usedTrainY
plotResult(usedTrainX, featureList, true_labels, 'True Scatter')

#%% 2.3 K-Means: finding cluster numbers
# Selecting the number of clusters with silhouette analysis on KMeans clustering
#   > http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# 

#%% 3.1 Mixture Gaussian CLustering and Visualization
# sklearn.mixture.GaussianMixture
#   > http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
from sklearn import mixture
# Setting data
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('all',trainX, trainY, testX, testY)

# Train models
gmm = mixture.GaussianMixture(n_components=7, covariance_type='full').fit(usedTrainX)
dpgmm = mixture.BayesianGaussianMixture(n_components=7,
                                        covariance_type='full').fit(usedTrainX)

# Visualization
plotResult(usedTrainX, featureList, gmm.predict(usedTrainX), 'Gaussian Mixture', wildMode = True)

# Gaussian Mixture Model Selection
# http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html