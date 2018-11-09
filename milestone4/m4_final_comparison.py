# -*- coding: utf-8 -*-
"""
[milestone 4 - Final Comparison]
    Compare all methods (at least the 3 milestones) you used throughout the semester 
    using 10 re-runs of a 10-fold cross-validation 
        and perform a suitable statistical test 
        to assess whether one of those performs significantly better than the others.

Models: 
    - Linear: ridge, linear SVM
    - Decision Tree
    - GP
    - NN
    - More classifiers in M1
    - Dimension reduction?
Evaluation Terms:
    - CV
    - Test
    - Sign Test
    - STD?
    - TIME!

Created on Wed Apr 25 14:50:59 2018

@author: remussn
"""
#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc, setUsedData
# Read data
# Remember to set path
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('batch',trainXCon, trainY, testXCon, testY)
#usedTrainXFull = trainX
#usedTestXFull = testX

# Binary
binary = False
if binary:
    type1 = 1
    type2 = 6
    usedTrainX = usedTrainX[(usedTrainY==type1)|(usedTrainY==type2)]
    usedTrainY = usedTrainY[(usedTrainY==type1)|(usedTrainY==type2)]    
    usedTestX = usedTestX[(usedTestY==type1)|(usedTestY==type2)]
    usedTestY = usedTestY[(usedTestY==type1)|(usedTestY==type2)]

del(trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon)

#%% 6. Try Different CLassifiers
import sklearn.linear_model as lm
from sklearn.svm import SVC
from sklearn import tree

maxIter = 1000
tolerance = 1e-3

svc = SVC()
svmLinear = SVC(kernel='linear')
ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.5, max_iter = maxIter)
logisitc = lm.LogisticRegression()
cartTree = tree.DecisionTreeClassifier()

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
gp = GaussianProcessClassifier(kernel=RBF(),multi_class='one_vs_one')

# Bagging
from sklearn.ensemble import BaggingClassifier
cartTree_bagging = BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0)

# Neural Network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(solver='lbfgs', alpha=1e-6,\
                            hidden_layer_sizes=[18,12], random_state=1)

modelList = [        
        ['ridge',ridge],
        ['RBF SVM', svc],
        ['Linear SVM', svmLinear],
        ['Logistic', logisitc],
        ['CART Decision Tree', cartTree],        
        ['CART bagging', cartTree_bagging],
        ['Gaussian Process', gp],
        ['Neural Network', nn]
        ]

import time
import numpy as np
from sklearn.model_selection import cross_val_score

usedModelList = []
for name, model in modelList:
    start_time = time.time()
    # Training / Fitting
    print('Name: ',name)
    theModel = model.fit(usedTrainX, usedTrainY)
    
    # Cross Validation
    CV = cross_val_score(model, usedTrainX, usedTrainY, cv=10).mean()
    print('average of 10VC: ', CV)
    
    # Testing
    score = model.score(usedTestX, usedTestY)
    print('Testing Score:\t', round(score,2)*100, '%')
    # print('Testing Score: ', score)
    
    # Running Time
    runTime = (time.time() - start_time)
    print("Time(s):\t", runTime)
    
    usedModelList.append({'name':name, 'CV':CV, 'test':score, 'time': runTime, 'model':theModel})

#%% Plot ACC
from plotAcc import plotAcc
if binary:
    plotAcc(usedModelList, 'Accuracies Using Different Models')
else:
    plotAcc(usedModelList, 'Accuracies Using Different Models')
    
#%% Plot Time
def plotTime(usedModelList, title, xlabel='Model', ylabel='Time(s)'):    
    import matplotlib.pyplot as plt
    n_groups = len(usedModelList)    
    names = [item['name'] for item in usedModelList]
    times = [item['time'] for item in usedModelList]
        
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.bar(index, times, bar_width, alpha=opacity, color='b',label='Time')
    # rects2 = plt.bar(index + bar_width, scores, bar_width, alpha=opacity, color='r', label='Test Accuracy')    
    # rects3 = plt.bar(index + 2*bar_width, times, bar_width, alpha=opacity, color='g', label='Time(s)')
    
    def autolabel(rects):    
    #Attach a text label above each bar displaying its height    
        for rect in rects:
            #height = round(rect.get_height()*10000)/100
            height = rect.get_height()
            print(height)
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    #'%d' % int(height),
                    str(int(height*100)/100),                
                    ha='center', va='bottom')

    autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)
    
    
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)    
    plt.title(title)
    plt.xticks(index + bar_width -0.35, names)    
    plt.legend();
      
    plt.tight_layout();   
    
    
    plt.show();       
    # plt.ylim(ymax=1)
    
plotTime(usedModelList, 'Time of Models')

#%% 5. Model Comparsion, using sign-test
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Set paras
usedModelCount = len(usedModelList)
reRunCount = 10
cvFoldCount= 10
winRecordArr = np.zeros([1,usedModelCount])# "sign array", records the winning times of each model

# 10 re-runs of a 10-fold cross-validation
for reRunIdx in range(reRunCount):
    print(str(reRunIdx+1)+' re-run')
    
    # Split
    cv = ShuffleSplit(n_splits=cvFoldCount, test_size=0.3, random_state=0)
    cvScoreRecordMat = np.zeros([usedModelCount,cvFoldCount])
    
    # CV score of each model
    for modelIdx in range(usedModelCount):
        cvScoreRecordMat[modelIdx,:] = cross_val_score(usedModelList[modelIdx]['model'], usedTrainX, usedTrainY, cv=cv)
    
    # Update winRecordArr
    cvMaxScoreIdx = np.argmax(cvScoreRecordMat,0)          # max of each column
    for modelIdx in range(usedModelCount):
        winRecordArr[0,modelIdx] += np.count_nonzero(cvMaxScoreIdx == modelIdx)    


print('Best Model: ' + usedModelList[np.argmax(winRecordArr)]['name'])
# array([[ 0.,  0., 20., 80.,  0.]])
# RBF, Matern, CART, SVM_RBF, SVM_POLY
import matplotlib.pyplot as plt
#plt.figure()
fig, ax = plt.subplots()
opacity = 0.80
rects = plt.bar(np.arange(usedModelCount), winRecordArr[0,:], alpha=opacity)
# plt.bar(np.arange(usedModelCount), ridge_coef_abs/np.linalg.norm(ridge_coef_abs),  alpha=opacity)
# plt.legend(['Linear SVM', 'ridge'])

plt.title('Sign test result of 10 re-runs 10-fold cross-validation')
plt.xticks(np.arange(usedModelCount),[item['name'] for item in usedModelList],size='small')
plt.xlabel('Models')
plt.ylabel('Wins')
plt.ylim([0,100])

for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
            str(int(height)),
            ha='center', va='bottom')
