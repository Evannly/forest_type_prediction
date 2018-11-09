#%% 0. Fundamental Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 1. Read dataset
# Read data
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
# Take a qucik look
# labeledData.head()
allX = labeledData.drop(['Id', 'Cover_Type'],axis=1)
allY = labeledData['Cover_Type']

discreteCols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]
continuousCols = [x for x in allX.columns if x not in discreteCols]

allXDis = allX[discreteCols]
allXCon = allX[continuousCols]

#%% 2. Preprosess
from sklearn import preprocessing
allXConScaled = preprocessing.scale(allXCon)*100
allXConScaled = allXConScaled - np.min(allXConScaled,0)
# https://www.cnblogs.com/chaosimple/p/4153167.html

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(allXConScaled)
allXConENC = enc.transform(allXConScaled).toarray()

allXENC = np.concatenate((allXConENC, np.array(allXDis)),axis=1)

# 2.3 Split
from sklearn.cross_validation import train_test_split
# trainX, testX, trainY, testY = train_test_split(allXENC, allY, random_state=1)
trainX, testX, trainY, testY = train_test_split(allXCon, allY, random_state=1)

# 2.4 Add Bias Dimension
# allXCount = len(allX[allX.columns[0]])
# allX['Bias'] = pd.Series(np.ones(allXCount), index=allX.index)

# testXDis = testX[discreteCols];     testXCon = testX[continuousCols]
# trainXDis = trainX[discreteCols];   trainXCon = trainX[continuousCols]

#%% 3. Experiment with regularizer coefficients
import sklearn.linear_model as lm
maxIter = 1000
tolerance = 1e-3
'''
alp_count = 30
alp_accuracy = np.zeros([1,alp_count])
for alp in np.linspace(0.0001,1,alp_count):
    ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=alp)
    ridge.fit(trainXCon, trainY)
    print('ALPHA: ', alp, 'ACC: ', ridge.score(testXCon, testY))
'''

#%% 4. Construct CLassifiers
# Choose model
# from sklearn import gaussian_process
# Gaussian = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
# GaussianProcessRegressor
# from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC;        svm = SVC(kernel='linear')
from sklearn.svm import LinearSVC;  svmLinear = LinearSVC()
from sklearn import tree;           cartTree = tree.DecisionTreeClassifier()

linear_square = lm.SGDClassifier(loss='squared_loss', penalty='none', max_iter=maxIter, tol=tolerance)
ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=0.5)
# ridgel1 = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance)
lasso = lm.SGDClassifier(loss='squared_loss', penalty='l1', max_iter=maxIter, tol=tolerance)
logisitc = lm.LogisticRegression()
# bayes = lm.BayesianRidge()


# Bagging
ridge_bagging = BaggingClassifier(ridge,
                            max_samples=0.7, max_features=1.0)
logisitc_bagging = BaggingClassifier(logisitc,
                            max_samples=0.7, max_features=1.0)
cartTree_bagging = BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0)
cartTree_Adaboost = AdaBoostClassifier(cartTree,
                            n_estimators=10,random_state=np.random.RandomState(1))

modelList = [
        ['linear_square',linear_square], 
        ['ridge',ridge], 
        # ['ridge bagging', ridge_bagging],
        # ['lasso',lasso], 
        # ['Bayes',bayes], 
        # ['Gaussian Proceee', Gaussian], 
        # ['SVM', svm], 
        ['Linear SVM', svmLinear],
        ['Logistic', logisitc],
        #['Logistic Bagging', logisitc_bagging]
        ['CART Decision Tree', cartTree],        
        ['CART bagging', cartTree_bagging],
        ['CART Adaboost', cartTree_Adaboost]        
        ]

import time
from sklearn.model_selection import cross_val_score
usedModelList = []
for name, model in modelList:
    start_time = time.time()
    # Training / Fitting
    print('Name: ',name)
    model.fit(trainX, trainY)
    
    # Cross Validation
    tenFoldCV = cross_val_score(model, trainX, trainY, cv=10)
    avgCV = np.mean(tenFoldCV)
    print('average of 10VC: ', avgCV)
    
    # Testing
    score = model.score(testX, testY)
    print('Testing Score:\t', round(score,4)*100, '%')
    print("Time(s):\t", (time.time() - start_time))
    usedModelList.append([name, avgCV, score])
    
xDrawPillar(usedModelList, 'CV and Test result, discretized features')
    
# def xVote(modelList):
def xDrawPillar(usedModelList, title):    
    n_groups = len(usedModelList);
    names = [item[0] for item in usedModelList]
    CVs   = [item[1] for item in usedModelList]
    scores= [item[2] for item in usedModelList]
    # means_men = (20, 35, 30, 35, 27)    
    # means_women = (25, 32, 34, 20, 25)    
         
    fig, ax = plt.pyplot.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.pyplot.bar(index, CVs, bar_width,alpha=opacity, color='b',label=    '10-fold Cross Validation')    
    rects2 = plt.pyplot.bar(index + bar_width, scores, bar_width,alpha=opacity,color='r',label='Test Accuracy')    
         
    plt.pyplot.xlabel('Models')    
    plt.pyplot.ylabel('Scores')    
    plt.pyplot.title(title)
    plt.pyplot.xticks(index + bar_width -0.15, names)
    # plt.ylim(0,40);    
    plt.pyplot.legend();    
      
    plt.pyplot.tight_layout();   
    plt.pyplot.show(); 
    ylim(ymax=1)