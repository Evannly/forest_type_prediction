# -*- coding: utf-8 -*-
"""
[milestone 3 - Dimensionality Reduction] 
Perform dimensionality reduction (PCA/SVD) 
    and visualize the data (incorporate the class leables in your visualization), 
    as well as use the new feature representation in a classifier/regression method you evaluated before. 

Compare the predictions results using The model evaluation strategy dervied in milestone xx.

[optional - Neural Network] 
Train and run a Neural Network. 
Evaluate the predictions using 10-fold cross-validation and a suitable error measure.

[optional - Efficiency] 
Compare the traning time and test time of the milestone 1, 2, and 3 methods 
    (plus task 4 - if your team did it). 
Use the average (or mode) runtime over 10 re-runs 
    and perform a suitable statistical test to assess whether one of those performs significantly better 
    than the others w.r.t. efficieny of training and test time.


Created on Mon Apr 16 17:58:54 2018

@author: remussn
"""

#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc, setUsedData
# Read data
# Remember to set path
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('all',trainXCon, trainY, testXCon, testY)

# Binary
binary = False
if binary:
    usedTrainY[list(usedTrainY<=3)] = 0; usedTrainY[list(usedTrainY > 3)] = 1;
    usedTestY[list(usedTestY<=3)] = 0;  usedTestY[list(usedTestY > 3)] = 1;

del(trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon)

#%% 2. PCA, SVD
# Doing PCA/SDV
# Record Running time and accuracy
from sklearn.decomposition import PCA, TruncatedSVD
import time
targetDim = 5
pca = PCA(n_components=targetDim)
start_time = time.time()
xPCA = pca.fit_transform(usedTrainX)
pcaFitTime = time.time() - start_time

svd = TruncatedSVD(n_components=targetDim)
start_time = time.time()
xSVD = svd.fit_transform(usedTrainX)
svdFitTime = time.time() - start_time

#%% 3. Autoencoder
# blog.csdn.net/marsjhao/article/details/68928486
# morvanzhou.github.io/tutorials/machine-learning/keras/2-6-autoencoder/
from keras.models import Model
from keras.layers import Dense, Input


inputLayer = Input(shape=(10,))
  
# Encoding Layers
if targetDim < 5:
    encoded = Dense(10, activation='relu')(inputLayer)  
    encoded = Dense(5, activation='relu')(encoded)
    encoder_output = Dense(targetDim)(encoded)  
    # Decoding Layers
    decoded = Dense(5, activation='relu')(encoder_output)  
    decoded = Dense(10, activation='tanh')(decoded)  
else:# targetDim~[5,10]
    encoded = Dense(10, activation='relu')(inputLayer)  
    encoder_output = Dense(targetDim)(encoded)  
    # Decoding Layers
    decoded = Dense(10, activation='tanh')(encoder_output)  
  
# Autoencoder
autoencoder = Model(inputs=inputLayer, outputs=decoded)
  
# Encoder
encoder = Model(inputs=inputLayer, outputs=encoder_output)
  
# Compile Autoencoder  
autoencoder.compile(optimizer='adam', loss='mse')  
  
# Training  
autoencoder.fit(usedTrainX, usedTrainX, epochs=60, batch_size=256, shuffle=True)

# Dimension Reduction Result
xAutoEncoder = encoder.predict(usedTrainX)

#%% 4. SVM CLassifier
from sklearn.svm import SVC
svcPCA = SVC().fit(xPCA, usedTrainY)
svcSVD = SVC().fit(xSVD, usedTrainY)
svcAUTO = SVC().fit(xAutoEncoder, usedTrainY)

print('ACC of PCA:'+str(svcPCA.score(pca.transform(usedTestX), usedTestY)))
print('ACC of SVD:'+str(svcSVD.score(svd.transform(usedTestX), usedTestY)))
print('ACC of AUTO:'+str(svcAUTO.score(encoder.predict(usedTestX), usedTestY)))
# print('ACC of AUTO on Training Data:'+str(svcAUTO.score(encoder.predict(usedTrainX), usedTrainY)))
#svcPCA = svc.fit(xPCA, usedTrainY)

#%% Visualization
def plotResult(X, y, title):    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    #ax.scatter(usedTrainX[featureList[0]], usedTrainX[featureList[1]], usedTrainX[featureList[2]],
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y)#, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(title)
    ax.dist = 12
    
#plotResult(xPCA, usedTrainY, 'PCA')
#plotResult(xSVD, usedTrainY, 'SVD')
#plotResult(xAutoEncoder, usedTrainY, 'AutoEncoder')