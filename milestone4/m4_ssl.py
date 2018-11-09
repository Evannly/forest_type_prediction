"""
[optional - Semi-Supervised Learning]
    Perform semi-supervised learning and compare your evaluations to a comparable method 
    (you want to achieve a fair compariosn) using the model evaluation strategy 
        dervied in milestones 1 and 2.

MORE: setting kernel hyperparameters(rbf/knn, gamma in rbf)        

[milestone 4 - Final Comparison]
    Compare all methods (at least the 3 milestones) you used throughout the semester 
    using 10 re-runs of a 10-fold cross-validation 
        and perform a suitable statistical test 
        to assess whether one of those performs significantly better than the others.
"""
#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc, setUsedData
# Read data
# Remember to set path
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('all',trainXCon, trainY, testXCon, testY)
#usedTrainXFull = trainX
#usedTestXFull = testX

# Binary
binary = False
if binary:
    type1 = 1
    type2 = 2
    usedTrainX = usedTrainX[(usedTrainY==type1)|(usedTrainY==type2)]
    usedTrainY = usedTrainY[(usedTrainY==type1)|(usedTrainY==type2)]    
    usedTestX = usedTestX[(usedTestY==type1)|(usedTestY==type2)]
    usedTestY = usedTestY[(usedTestY==type1)|(usedTestY==type2)]

del(trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon)

#%% 2. Label Propagation
#http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-versus-svm-iris-py
#http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-py

# Create unlabeled data
import numpy as np
rng = np.random.RandomState(0)
# y_10: 10% data are labeled
y_10 = np.copy(usedTrainY)
y_10[rng.rand(len(usedTrainY)) < 0.9] = -1
y_30 = np.copy(usedTrainY)
y_30[rng.rand(len(usedTrainY)) < 0.7] = -1
y_50 = np.copy(usedTrainY)
y_50[rng.rand(len(usedTrainY)) < 0.5] = -1
y_70 = np.copy(usedTrainY)
y_70[rng.rand(len(usedTrainY)) < 0.3] = -1

# Label Propagation
from sklearn.semi_supervised import label_propagation
predY10 = label_propagation.LabelSpreading().fit(usedTrainX, y_10).transduction_
predY30 = label_propagation.LabelSpreading().fit(usedTrainX, y_30).transduction_
predY50 = label_propagation.LabelSpreading().fit(usedTrainX, y_50).transduction_
predY70 = label_propagation.LabelSpreading().fit(usedTrainX, y_70).transduction_

#%% 3. Predict(semi-supervised)
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.model_selection import cross_val_score
cartTree = tree.DecisionTreeClassifier()
cartTree_bagging = BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0)
#svmLinear = SVC(kernel='linear')

#import sklearn.linear_model as lm
#maxIter = 1000
#tolerance = 1e-3
#ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.5, max_iter = maxIter)

yList = [predY10, predY30, predY50, predY70, usedTrainY]
yListName = ['10% labeled', '30% labeled','50% labeled', '70% labeled', 'All data']
semiRecord = []
for yIdx, y in enumerate(yList):    
    CV = cross_val_score(BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0), usedTrainX, y, cv=5).mean()    
#    CV = cross_val_score(ridge, usedTrainX, y, cv=5).mean()    
    tree = cartTree_bagging.fit(usedTrainX, y)
    tree.fit(usedTrainX, y)
    testScore = tree.score(usedTestX, usedTestY)
    
    print(yListName[yIdx])
    print('CV:\t{}'.format(CV))
    print('Test:\t{}'.format(testScore))
    
    semiRecord.append({'name':yListName[yIdx], 'CV':CV, 'test': testScore})

#%% 4. Predict(batch data)
from sklearn.cross_validation import train_test_split
x_batch_10 = usedTrainX[y_10!=-1];  y_batch_10 = usedTrainY[y_10!=-1]
x_batch_30 = usedTrainX[y_30!=-1];  y_batch_30 = usedTrainY[y_30!=-1]
x_batch_50 = usedTrainX[y_50!=-1];  y_batch_50 = usedTrainY[y_50!=-1]
x_batch_70 = usedTrainX[y_70!=-1];  y_batch_70 = usedTrainY[y_70!=-1]
#x_batch_10, _, y_batch_10,_  = train_test_split(usedTrainX, usedTrainY, test_size = 0.9, random_state=1)
#x_batch_30, _, y_batch_30,_  = train_test_split(usedTrainX, usedTrainY, test_size = 0.7, random_state=1)
#x_batch_50, _, y_batch_50,_  = train_test_split(usedTrainX, usedTrainY, test_size = 0.5, random_state=1)    
#x_batch_70, _, y_batch_70,_  = train_test_split(usedTrainX, usedTrainY, test_size = 0.3, random_state=1)
    
batchList = [[x_batch_10, y_batch_10], [x_batch_30, y_batch_30], [x_batch_50, y_batch_50], [x_batch_70, y_batch_70]]
batchListName = ['10% data', '30% data','50% data', '70% data']

batchRecord = []
for idx, (x,y) in enumerate(batchList):
#    CV = cross_val_score(ridge, x, y, cv=5).mean()
    CV = cross_val_score(BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0), x, y, cv=5).mean()    
    tree = cartTree_bagging.fit(x, y)
    #ridge.fit(x, y)
    testScore = tree.score(usedTestX, usedTestY)
    
    print(batchListName[idx])
    print('CV:\t{}'.format(CV))
    print('Test:\t{}'.format(testScore))
    
    batchRecord.append({'name':batchListName[idx], 'CV':CV, 'test': testScore})

#%% Visualize
from plotAcc import plotAcc
if binary:
    plotAcc(batchRecord+semiRecord, 'Accuracies Using Different Data (Soil Types {} & {})'.format(type1,type2), xlabel='Data')
else:
    plotAcc(batchRecord+semiRecord, 'Accuracies Using Different Data (All 7 Soil Types)', xlabel='Data')