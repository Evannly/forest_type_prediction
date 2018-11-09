#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc, setUsedData
# Read data
# Remember to set path
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
trainBatchSize = 500
testBatchSize = 100

usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('batch',trainXCon, trainY, testXCon, testY, trainBatchSize, testBatchSize)

#%% 2. Gaussian Procecss
# Hyperparameters are automatically optimized!
if 'usedModelList' in locals():
    del(usedModelList)
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn import tree;           cartTree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
cartTree_bagging = BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0)

kernelList = [
        ['RBF',RBF()],
        ['Matern',Matern()],  
        ]

# List of Gaussian Processes
gpList = []
for name, gp_kernel in kernelList:
    gpc = GaussianProcessClassifier(kernel=gp_kernel,multi_class='one_vs_one',n_jobs=-1)
    #gpc = GaussianProcessClassifier(kernel=gp_kernel,multi_class='one_vs_rest',n_jobs=-1)
    gpList.append(['GP_'+name,gpc])

# List of all models
# Here we add all GP into usedModelList
# Add decision tree as a comparsion
gpList.append(['CART bagging', cartTree_bagging])
gpList.append(['SVC_RBF', SVC(C = 400, gamma = 0.01)])
gpList.append(['SVC_POLY', SVC(kernel='poly',degree=1.789)])
usedModelList = []
import time
from sklearn.model_selection import cross_val_score
for name, gpc in gpList:
    start_time = time.time()
    
    # Training / Fitting
    print('\nName: ',name)
    gpc.fit(usedTrainX, usedTrainY)
    
    # Cross Validation
    tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=5, n_jobs=-1)
    print('Average of 10VC: ', round(tenFoldCV.mean(),4)*100, '%')
    print('Std of 10VC: ', tenFoldCV.std())
    
    # Testing
    score = gpc.score(usedTestX, usedTestY)
    print('Testing Score:\t', round(score,4)*100, '%')
    
    # Running Time
    print("Time(s):\t", round((time.time() - start_time)*100)/100)
    
    # Add to usedModelList
    usedModelList.append([name, tenFoldCV.mean(), score, gpc, tenFoldCV.std()])
    
#%% 4. Model Comparsion, using Accuracy
from plotAcc import plotAcc
plotAcc(usedModelList, '')

#%% 5. Model Comparsion, using sign-test
from sklearn.model_selection import ShuffleSplit, cross_val_score

# Set paras
usedModelCount = len(usedModelList)
reRunCount = 3
cvFoldCount= 3
winRecordArr = np.zeros([1,usedModelCount])# "sign array", records the winning times of each model

# 10 re-runs of a 10-fold cross-validation
for reRunIdx in range(reRunCount):
    print(str(reRunIdx+1)+' re-run')
    
    # Split
    cv = ShuffleSplit(n_splits=cvFoldCount, test_size=0.3, random_state=0)
    cvScoreRecordMat = np.zeros([usedModelCount,cvFoldCount])
    
    # CV score of each model
    for modelIdx in range(usedModelCount):
        cvScoreRecordMat[modelIdx,:] = cross_val_score(usedModelList[modelIdx][3], usedTrainX, usedTrainY, cv=cv, n_jobs=-1)
    
    # Update winRecordArr
    cvMaxScoreIdx = np.argmax(cvScoreRecordMat,0)          # max of each column
    for modelIdx in range(usedModelCount):
        winRecordArr[0,modelIdx] += np.count_nonzero(cvMaxScoreIdx == modelIdx)    


print('Best Model: ' + usedModelList[np.argmax(winRecordArr)][0])
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
plt.xticks(np.arange(usedModelCount),[item[0] for item in usedModelList],size='small')
plt.xlabel('Models')
plt.ylabel('Wins')
plt.ylim([0,100])

for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            str(int(height)),
            ha='center', va='bottom')