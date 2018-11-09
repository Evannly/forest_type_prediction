# -*- coding: utf-8 -*-
"""
[milestone 4 - Final Comparison]
    Compare all methods (at least the 3 milestones) you used throughout the semester 
    using 10 re-runs of a 10-fold cross-validation 
        and perform a suitable statistical test 
        to assess whether one of those performs significantly better than the others.

Using Friedman test and nemenyi_multitest as statistical test
http://tec.citius.usc.es/stac/doc/#
https://github.com/citiususc/stac
https://stats.stackexchange.com/questions/246719/friedman-test-and-post-hoc-test-for-python/267250?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

https://stackoverflow.com/questions/31195941/what-is-the-correct-way-of-passing-parameters-to-stats-friedmanchisquare-based-o
https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.friedmanchisquare.html

Created on Thu May  3 17:00:22 2018

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
        [ 'Neural Network', nn]
        ]

import time
import numpy as np
from sklearn.model_selection import cross_val_score

usedModelList = []
rerunNum = 1

for name, model in modelList:
    training_time_total = 0
    test_time_total = 0
    for rerun in range(rerunNum):
        training_start_time = time.time()
        # Training / Fitting
        print('Name: ',name)
        theModel = model.fit(usedTrainX, usedTrainY)
        training_time_total += time.time() - training_start_time
        
        # Cross Validation
        # CV = cross_val_score(model, usedTrainX, usedTrainY, cv=10).mean()
        # print('average of 10VC: ', CV)
        
        # Testing
        test_start_time = time.time()
        score = model.score(usedTestX, usedTestY)
        print('Testing Score:\t', round(score,2)*100, '%')
        # print('Testing Score: ', score)
        
        # Running Time
        #runTime = (time.time() - start_time)
        test_time_total += time.time() - test_start_time
        # print("Time(s):\t", test)
        
        # usedModelList.append({'name':name, 'CV':CV, 'test':score, 'training_time': training_time_total,'test_time':test_time_total, 'model':theModel})        
    usedModelList.append({'name':name, 'test':score, 'training_time': training_time_total,'test_time':test_time_total, 'model':theModel})

#%% Plot ACC
#from plotAcc import plotAcc
#if binary:
#    plotAcc(usedModelList, 'Accuracies Using Different Models')
#else:
#    plotAcc(usedModelList, 'Accuracies Using Different Models')
    
#%% Plot Time(train&test)
def plotTime(usedModelList, title, xlabel='Model', ylabel='Time(s)'):    
    import matplotlib.pyplot as plt
    n_groups = len(usedModelList)    
    names = [item['name'] for item in usedModelList]
    training_times = [item['training_time'] for item in usedModelList]
    test_times = [item['test_time'] for item in usedModelList]
        
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.bar(index, training_times, bar_width, alpha=opacity, color='b',label='Training Time')
    rects2 = plt.bar(index + bar_width, test_times, bar_width, alpha=opacity, color='r', label='Test Time')    
    # rects3 = plt.bar(index + 2*bar_width, times, bar_width, alpha=opacity, color='g', label='Time(s)')
    
    def autolabel(rects):    
    #Attach a text label above each bar displaying its height    
        for rect in rects:
            #height = round(rect.get_height()*10000)/100
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                    #'%d' % int(height),
                    str(int(height*100)/100),                
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    # autolabel(rects3)
    
    
    plt.xlabel(xlabel)    
    plt.ylabel(ylabel)    
    plt.title(title)
    plt.xticks(index + bar_width -0.35, names)    
    plt.legend();
      
    plt.tight_layout();   
    
    
    plt.show();       
    # plt.ylim(ymax=1)
    
plotTime(usedModelList, 'Total 10-rerun Time of Models')


#%% 5. Model Statistical Test: CV
from nonparametric_tests import friedman_test, nemenyi_multitest
# tec.citius.usc.es/stac/doc/stac.nonparametric_tests.friedman_test.html#stac.nonparametric_tests.friedman_test
# input: array([clfNum, scores]), i.e. each row is scores of a classifier
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Set paras
reRunCount = 10
cvFoldCount= 5
usedModelCount = len(usedModelList)
cvScoreRecordMat = np.zeros([usedModelCount, reRunCount*cvFoldCount])
# winRecordArr = np.zeros([1,usedModelCount])# "sign array", records the winning times of each model

# 10 re-runs of a 10-fold cross-validation
for reRunIdx in range(reRunCount):
    print(str(reRunIdx+1)+' re-run')
    
    # Split
    cv = ShuffleSplit(n_splits=cvFoldCount, test_size=0.3, random_state=0)
    
    # CV score of each model
    for modelIdx in range(usedModelCount):
        cvScoreRecordMat[modelIdx,reRunIdx*cvFoldCount:(reRunIdx+1)*cvFoldCount] = cross_val_score(usedModelList[modelIdx]['model'], usedTrainX, usedTrainY, cv=cv)

#%% 5. Model Statistical Test: Test
# Friedman Test    
_, _, friedman_rankings, _ = friedman_test(1 - cvScoreRecordMat)

# Nemenyi Test
# https://gist.github.com/garydoranjr/5016455
from nemenyi import critical_difference
alpha = 0.10
CD = critical_difference(pvalue = alpha, models = usedModelCount, datasets = reRunCount*cvFoldCount)

# Plot
# https://matplotlib.org/gallery/statistics/errorbar_features.html#sphx-glr-gallery-statistics-errorbar-features-py
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(friedman_rankings, np.arange(1,usedModelCount+1), xerr=CD, fmt='o')
plt.yticks(np.arange(1,usedModelCount+1), [model['name'] for model in usedModelList])
plt.xlabel('average ranking')
plt.ylabel('models')
plt.title('Friedman average rankings and Nemenyi critical values of models, confidence level = {}'.format(1-alpha))