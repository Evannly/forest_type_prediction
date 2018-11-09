# -*- coding: utf-8 -*-

"""
[optional - Neural Network] Train and run a Neural Network.
Evaluate the predictions using 10-fold cross-validation and a suitable error measure.
"""

import pandas as pd
from preprocessing import preproc, setUsedData

dataset = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(dataset)
trainBatchSize = 1000
testBatchSize = 300

usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('batch',trainX, trainY, testX, testY, trainBatchSize, testBatchSize)

# 2. neural network
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import ShuffleSplit, cross_val_score

left = 2
right = 20
i = 1
acc = np.zeros((right-left+1),dtype=int)
# for node in range(left,right,20):
# for layer in range(left,right):

# nodes = 2
# step = 4
# for layers in range(6,15,4):
    # mountain shape
    # hidden_layer = np.concatenate([np.arange(nodes,layers,step),np.arange(layers,nodes-1,-step)])

# nodes = 10
# for layers in range(1,15):
    # flat shape hidden layer
    # hidden_layer = nodes*np.ones([layers],dtype=int)

score = np.zeros((25,25,25))
for nodes1 in range(2,20):
    for nodes2 in range(2,nodes1):
        for nodes3 in range(2,nodes2):
            # hidden_layer = [10,6]
            hidden_layer = [nodes1,nodes2,nodes3]

            # print(hidden_layer)
            clf = MLPClassifier(solver='lbfgs', alpha=1e-6,\
                            hidden_layer_sizes=hidden_layer, random_state=1)
            clf.fit(usedTrainX, usedTrainY)

            # result = clf.predict(testX)
            # acc = 1-np.count_nonzero(testY-result)/len(result)
            # score[nodes1] = clf.score(testX,testY)
            # score[nodes1] = acc
            # print(acc)
            # print(nodes1)

            # cross validate
            cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
            cross_score = cross_val_score(clf, usedTrainX, usedTrainY, cv=cv)
            avg = np.mean(cross_score)
            score[nodes1,nodes2,nodes3] = avg
    print(nodes1)
print(np.max(score),np.where(score==np.max(score)))


# 3. 10-fold cross-validation

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

clf = MLPClassifier(solver='lbfgs', alpha=1e-6,\
                    hidden_layer_sizes=(18,12), random_state=1)
clf.fit(usedTrainX, usedTrainY)
cross_score = cross_val_score(clf, usedTrainX, usedTrainY, cv=cv)
avg = np.mean(cross_score)


