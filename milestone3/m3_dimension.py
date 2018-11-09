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
usedTrainXFull = trainX
usedTestXFull = testX


# Binary
binary = False
if binary:
    usedTrainXFull = usedTrainXFull[(usedTrainY==1)|(usedTrainY==6)]
    usedTestXFull = usedTestXFull[(usedTestY==1)|(usedTestY==6)]
    usedTrainX = usedTrainX[(usedTrainY==1)|(usedTrainY==6)]
    usedTrainY = usedTrainY[(usedTrainY==1)|(usedTrainY==6)]    
    usedTestX = usedTestX[(usedTestY==1)|(usedTestY==6)]
    usedTestY = usedTestY[(usedTestY==1)|(usedTestY==6)]
    #usedTrainY[list(usedTrainY<=3)] = 0; usedTrainY[list(usedTrainY > 3)] = 1;
    #usedTestY[list(usedTestY<=3)] = 0;  usedTestY[list(usedTestY > 3)] = 1;

del(trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon)

#%% 2. Universial Function
def encoding(targetDim, encoderType, X):    
    if encoderType == 'PCA':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=targetDim)
        return pca.fit_transform(X), pca
    elif encoderType == 'SVD':
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=targetDim)
        return svd.fit_transform(usedTrainX), svd
    elif encoderType == 'AutoEncoder':
        from keras.models import Model
        from keras.layers import Dense, Input
        
        # Input Layer
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
        autoencoder.fit(usedTrainX, usedTrainX, epochs=150, batch_size=256, shuffle=True, verbose=0)
        
        # Dimension Reduction Result
        return encoder.predict(usedTrainX), encoder
    elif encoderType == 'AutoEncoderFull':
        from keras.models import Model
        from keras.layers import Dense, Input
        
        # Input Layer
        inputLayer = Input(shape=(54,))
          
        # Encoding Layers
        encoded = Dense(54, activation='relu')(inputLayer)  
        encoded = Dense(25, activation='relu')(encoded)
        encoded = Dense(10, activation='relu')(encoded)
        encoder_output = Dense(targetDim)(encoded)  
        # Decoding Layers
        decoded = Dense(10, activation='relu')(encoder_output)
        decoded = Dense(25, activation='relu')(decoded)  
        decoded = Dense(54, activation='tanh')(decoded)            
        # Autoencoder
        autoencoder = Model(inputs=inputLayer, outputs=decoded)          
        # Encoder
        encoder = Model(inputs=inputLayer, outputs=encoder_output)          
        # Compile Autoencoder  
        autoencoder.compile(optimizer='adam', loss='mse')          
        # Training  
        autoencoder.fit(usedTrainX, usedTrainX, epochs=200, batch_size=256, shuffle=True, verbose=0)        
        # Dimension Reduction Result
        return encoder.predict(usedTrainX), encoder

#%% 
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

CVRecord = np.zeros(shape=(10,3))
testRecord = np.zeros(shape=(10,3))
for targetDim in range(1,10):
    print('Current Dimension: '+str(targetDim))
    xPCA, pca = encoding(targetDim, 'PCA', usedTrainX)
    xSVD, svd = encoding(targetDim, 'SVD', usedTrainX)
    xAutoEncoder, encoder = encoding(targetDim, 'AutoEncoder', usedTrainX)
    #xAutoEncoderFull, encoderFull = encoding(targetDim, 'AutoEncoder', usedTrainXFull)            
    
    CVRecord[targetDim-1, 0] = cross_val_score(SVC(), xPCA, usedTrainY, cv=10).mean()
    CVRecord[targetDim-1, 1] = cross_val_score(SVC(), xSVD, usedTrainY, cv=10).mean()
    CVRecord[targetDim-1, 2] = cross_val_score(SVC(), xAutoEncoder, usedTrainY, cv=10).mean()
    #CVRecord[targetDim-1, 3] = cross_val_score(SVC(), xAutoEncoderFull, usedTrainY, cv=10).mean()
    
    svcPCA = SVC().fit(xPCA, usedTrainY)
    svcSVD = SVC().fit(xSVD, usedTrainY)
    svcAUTO = SVC().fit(xAutoEncoder, usedTrainY)
    #svcAUTOFull = SVC().fit(xAutoEncoderFull, usedTrainY)
    
    testRecord[targetDim-1, 0] = svcPCA.score(pca.transform(usedTestX), usedTestY)
    testRecord[targetDim-1, 1] = svcSVD.score(svd.transform(usedTestX), usedTestY)
    testRecord[targetDim-1, 2] = svcAUTO.score(encoder.predict(usedTestX), usedTestY)
    #testRecord[targetDim-1, 3] = svcAUTOFull.score(encoderFull.predict(usedTestX), usedTestY)

# Accuracy with all 10 continuous features
# CV
CVFullData = cross_val_score(SVC(), usedTrainX, usedTrainY, cv=10).mean()
CVRecord[9, 0] = CVFullData
CVRecord[9, 1] = CVFullData
CVRecord[9, 2] = CVFullData
# Test Acc
svcFullData = SVC().fit(usedTrainX, usedTrainY)
fullDataScore = svcFullData.score(usedTestX, usedTestY)
testRecord[9, 0] = fullDataScore
testRecord[9, 1] = fullDataScore
testRecord[9, 2] = fullDataScore


# Accuracy with full features (Continuous & discrete)
#svcFullFeature = SVC().fit(usedTrainXFull, usedTrainY)
#testRecord[9, 3] = svcFullFeature.score(usedTestXFull, usedTestY)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(121)
plt.plot(CVRecord)
plt.legend(['PCA','SVD','AutoEncoder','AutoEncoder(All Features)'])
plt.xticks( np.arange(10), np.arange(10)+1 )
plt.xlabel('Dimension (dim = 10 means no dim reduction)')
plt.ylabel('Accuracy')
if binary:
    plt.title('10-Fold CV Accuracy of SVM (Soil Types 1 & 6)')
else:
    plt.title('10-Fold CV Accuracy of SVM (All 7 Soil Types)')        


#plt.figure()
plt.subplot(122)
plt.plot(testRecord)
plt.legend(['PCA','SVD','AutoEncoder','AutoEncoder(All Features)'])
plt.xticks( np.arange(10), np.arange(10)+1 )
plt.xlabel('Dimension (dim = 10 means no dim reduction)')
plt.ylabel('Accuracy')
if binary:
    plt.title('Test Accuracy of SVM (Soil Types 1 & 6)')
else:
    plt.title('Test Accuracy of SVM (All 7 Soil Types)')        

#%% 2. PCA, SVD
## Doing PCA/SDV
## Record Running time and accuracy
#from sklearn.decomposition import PCA, TruncatedSVD
## import time
#targetDim = 3
#pca = PCA(n_components=targetDim)
##start_time = time.time()
#xPCA = pca.fit_transform(usedTrainX)
##pcaFitTime = time.time() - start_time
#
#svd = TruncatedSVD(n_components=targetDim)
##start_time = time.time()
#xSVD = svd.fit_transform(usedTrainX)
##svdFitTime = time.time() - start_time
#
##%% 3. Autoencoder
## blog.csdn.net/marsjhao/article/details/68928486
## morvanzhou.github.io/tutorials/machine-learning/keras/2-6-autoencoder/
#from keras.models import Model
#from keras.layers import Dense, Input
#
#
#inputLayer = Input(shape=(10,))
#  
## Encoding Layers
#if targetDim < 5:
#    encoded = Dense(10, activation='relu')(inputLayer)  
#    encoded = Dense(5, activation='relu')(encoded)
#    encoder_output = Dense(targetDim)(encoded)  
#    # Decoding Layers
#    decoded = Dense(5, activation='relu')(encoder_output)  
#    decoded = Dense(10, activation='tanh')(decoded)  
#else:# targetDim~[5,10]
#    encoded = Dense(10, activation='relu')(inputLayer)  
#    encoder_output = Dense(targetDim)(encoded)  
#    # Decoding Layers
#    decoded = Dense(10, activation='tanh')(encoder_output)  
#  
## Autoencoder
#autoencoder = Model(inputs=inputLayer, outputs=decoded)
#  
## Encoder
#encoder = Model(inputs=inputLayer, outputs=encoder_output)
#  
## Compile Autoencoder  
#autoencoder.compile(optimizer='adam', loss='mse')  
#  
## Training  
#autoencoder.fit(usedTrainX, usedTrainX, epochs=100, batch_size=256, shuffle=True)
#
## Dimension Reduction Result
#xAutoEncoder = encoder.predict(usedTrainX)
#
##%% 4. SVM CLassifier
#from sklearn.svm import SVC
#svcPCA = SVC().fit(xPCA, usedTrainY)
#svcSVD = SVC().fit(xSVD, usedTrainY)
#svcAUTO = SVC().fit(xAutoEncoder, usedTrainY)
#
#print('ACC of PCA:'+str(svcPCA.score(pca.transform(usedTestX), usedTestY)))
#print('ACC of SVD:'+str(svcSVD.score(svd.transform(usedTestX), usedTestY)))
#print('ACC of AUTO:'+str(svcAUTO.score(encoder.predict(usedTestX), usedTestY)))
## print('ACC of AUTO on Training Data:'+str(svcAUTO.score(encoder.predict(usedTrainX), usedTrainY)))
##svcPCA = svc.fit(xPCA, usedTrainY)

#%% Visualization (separately)
def plotResult(X, y, title, fig):    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y, label=list(y))#, edgecolor='k')    
    # ax.legend()

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.dist = 12
    
if binary:    
    plotResult(xPCA, usedTrainY, 'PCA (Soil Types 1 & 6)')
    plotResult(xSVD, usedTrainY, 'PCA (Soil Types 1 & 6)')
    plotResult(xAutoEncoder, usedTrainY, 'PCA (Soil Types 1 & 6)')
else:
    plotResult(xPCA, usedTrainY, 'PCA (All 7 Soil Types)')
    plotResult(xSVD, usedTrainY, 'SVD (All 7 Soil Types)')
    plotResult(xAutoEncoder, usedTrainY, 'AutoEncoder (All 7 Soil Types)')

#%% Visualization (together)
def plotResults(Xs, y, titles):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    #fig = plt.figure(figsize=(4, 3))
    fig = plt.figure()
    for idx in range(len(Xs)):        
        X = Xs[idx]
        title = titles[idx]
        #plt.subplot(1,len(Xs),idx+1)
        # fig = plt.subplot(131)
        #ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        #ax = Axes3D(fig)
        ax = fig.add_subplot(1, len(Xs), idx+1, projection='3d')
        ax.scatter(X[:,0],X[:,1],X[:,2],c=y, label=list(y))#, edgecolor='k')    
    
#        ax.w_xaxis.set_ticklabels([])
#        ax.w_yaxis.set_ticklabels([])
#        ax.w_zaxis.set_ticklabels([])
        ax.set_title(title)
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Dim 3')
        ax.dist = 12
    
if binary:
    plotResults([xPCA,xSVD,xAutoEncoder], usedTrainY, ['PCA (Soil Types 1 & 6)','SVD (Soil Types 1 & 6)','Autoencoder (Soil Types 1 & 6)'])
else:    
    plotResult(xAutoEncoder, usedTrainY, 'AutoEncoder (All 7 Soil Types)')
