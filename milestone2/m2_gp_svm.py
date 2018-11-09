# -*- coding: utf-8 -*-
"""
CSE 517A Machine Learning
Application Project, Milestone 2

# [milestone 2 - Gaussian Process] Train and run Gaussian Processes.
    Evaluate and compare the predictions using at least two differnt kernels 
    via 10-fold cross-validation with a suitable error measure 
    (we recommend negative log predictiv density as it takes the predictive uncertainty into account).

# [optional - Support Vector Machine]
    Train and run a kernel Support Vector Machine.
    Evaluate the predictions using 10-fold cross-validation and a suitable error measure.
    
# [optional - Model Evaluation] 
    Compare at 2 differnet methods using 10 re-runs of a 10-fold cross-validation 
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
trainBatchSize = 500
testBatchSize = 100

usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle = setUsedData('batch',trainXCon, trainY, testXCon, testY, trainBatchSize, testBatchSize)

#%% 2. Gaussian Procecss
# Hyperparameters are automatically optimized!
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, Exponentiation
from sklearn import tree;           cartTree = tree.DecisionTreeClassifier()
from sklearn.ensemble import BaggingClassifier
cartTree_bagging = BaggingClassifier(cartTree,
                            max_samples=0.7, max_features=1.0)
# Full kernel list:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
kernelList = [
        ['RBF',RBF()],
        ['Matern',Matern()],
        #['Exponentiation',Exponentiation()],
        #['Constant', ConstantKernel()]       
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
usedModelList = []
import time
from sklearn.model_selection import cross_val_score
for name, gpc in gpList:
    start_time = time.time()
    
    # Training / Fitting
    print('\nName: ',name)
    gpc.fit(usedTrainX, usedTrainY)
    
    # Cross Validation
    #tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=10, scoring='neg_log_loss', n_jobs=-1)
    tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=5, n_jobs=-1)
    #avgCV = np.mean(tenFoldCV)
    print('Average of 10VC: ', round(tenFoldCV.mean(),4)*100, '%')
    print('Std of 10VC: ', tenFoldCV.std())
    
    # Testing
    score = gpc.score(usedTestX, usedTestY)
    print('Testing Score:\t', round(score,4)*100, '%')
    
    # Running Time
    print("Time(s):\t", round((time.time() - start_time)*100)/100)
    
    # Add to usedModelList
    usedModelList.append([name, tenFoldCV.mean(), score, gpc, tenFoldCV.std()])
    

#%% Measuring Training+CV+Testing Time for models in usedModelList
for name,m,s, model,std in usedModelList:
    start_time = time.time()
    
    # Training / Fitting
    print('\nName: ',name)
    gpc.fit(usedTrainX, usedTrainY)
    
    # Cross Validation
    #tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=10, scoring='neg_log_loss', n_jobs=-1)
    tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=10, n_jobs=-1)
    #avgCV = np.mean(tenFoldCV)
    print('Average of 10VC: ', round(tenFoldCV.mean()*100,2)/100, '%')
    print('Std of 10VC: ', tenFoldCV.std())
    
    # Testing
    score = gpc.score(usedTestX, usedTestY)
    print('Testing Score:\t', round(score*100,2)/100, '%')
    
    # Running Time
    print("Time(s):\t", round((time.time() - start_time)*100)/100)
    
    # Add to usedModelList
    # usedModelList.append([name, tenFoldCV.mean(), score, gpc, tenFoldCV.std()])
    
#%% 3.0 SVM(BRF, Poly) Find Best SVC Hyperparameters by Telescope Search + Grid Search
# Use precomputed result
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
start_time = time.time()
svc_rbf_best_para_X = dict(C = 400, gamma = 0.01)
svc_rbf_C_range_X = np.linspace(svc_rbf_best_para_X['C']*0.1,svc_rbf_best_para_X['C']*10,20)
svc_rbf_gamma_range_X = np.linspace(svc_rbf_best_para_X['gamma']*0.1,svc_rbf_best_para_X['gamma']*10,20)
param_grid = dict(gamma=svc_rbf_gamma_range_X, C=svc_rbf_C_range_X)
grid = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid.fit(usedTrainX, usedTrainY)
print('3rd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))
print("Time(s):\t", round((time.time() - start_time)*100)/100)

# Choose best model/parameter
svc_rbf_best = grid.best_estimator_
svc_rbf_score = svc_rbf_best.score(usedTestX,usedTestY) # test accuracy
tenFoldCV = cross_val_score(svc_rbf_best, usedTrainX, usedTrainY, cv=10, n_jobs=-1)
#avgCV = np.mean(tenFoldCV)
usedModelList.append(['svm_rbf', tenFoldCV.mean(), svc_rbf_score, svc_rbf_best, tenFoldCV.std()])
print('3rd search, test acc = ' + str(svc_rbf_score))


start_time = time.time()
coef0 = [1];    # Fix coef0
svc_ploy_best_para_X = dict(degree = 5)
svc_poly_degree_range_X = np.linspace(svc_ploy_best_para_X['degree']*0.1,svc_ploy_best_para_X['degree']*5,20)
param_grid = dict(coef0=coef0, degree=svc_poly_degree_range_X)
grid = GridSearchCV(SVC(kernel='poly'), param_grid=param_grid, cv=StratifiedKFold(n_splits=5),n_jobs=-1)
grid.fit(usedTrainX, usedTrainY)

# Choose best model/parameter
svc_poly_best = grid.best_estimator_
svc_poly_score = svc_poly_best.score(usedTestX,usedTestY)
tenFoldCV = cross_val_score(svc_poly_best, usedTrainX, usedTrainY, cv=10, n_jobs=-1)
#avgCV = np.mean(tenFoldCV)
usedModelList.append(['svm_poly',  tenFoldCV.mean(), svc_poly_score, svc_poly_best,  tenFoldCV.std()])
print('3rd search, test acc = ' + str(svc_poly_score))
print("Time(s):\t", (time.time() - start_time))



#%% 3.1 SVM(RBF): Find Best SVC Hyperparameters by Telescope Search + Grid Search (RBF Kernel)
# http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/svm/plot_svm_parameters_selection.html
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
start_time = time.time()

# 1st Search
svc_rbf_C_range_1 = 10. ** np.linspace(-3,8,12)
svc_rbf_gamma_range_1 = 10. ** np.linspace(-5,4,10)
param_grid = dict(gamma=svc_rbf_gamma_range_1, C=svc_rbf_C_range_1)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_rbf_best_para_1 = grid.best_params_
print('1st search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 2nd Search
svc_rbf_C_range_2 = np.linspace(svc_rbf_best_para_1['C']*0.1,svc_rbf_best_para_1['C']*10,10)
svc_rbf_gamma_range_2 = np.linspace(svc_rbf_best_para_1['gamma']*0.1,svc_rbf_best_para_1['gamma']*10,10)
param_grid = dict(gamma=svc_rbf_gamma_range_2, C=svc_rbf_C_range_2)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_rbf_best_para_2 = grid.best_params_
print('2nd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 3rd Search(save time)
"""
svc_rbf_C_range_3 = np.linspace(svc_rbf_best_para_2['C']*0.5,svc_rbf_best_para_2['C']*1.5,10)
svc_rbf_gamma_range_3 = np.linspace(svc_rbf_best_para_2['gamma']*0.5,svc_rbf_best_para_2['gamma']*1.5,10)
param_grid = dict(gamma=svc_rbf_gamma_range_3, C=svc_rbf_C_range_3)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
"""
svc_rbf_best_para_3 = grid.best_params_
print('3rd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))
print("Time(s):\t", (time.time() - start_time))

# score: 0.6833333333333333
# {'C': 733.3333333333334, 'gamma': 0.00575}
# 25 min

# Choose best model/parameter
svc_rbf_best = grid.best_estimator_
svc_rbf_score = svc_rbf_best.score(usedTestX,usedTestY) # test accuracy
tenFoldCV = cross_val_score(svc_rbf_best, usedTrainX, usedTrainY, cv=10, n_jobs=-1)
#avgCV = np.mean(tenFoldCV)
usedModelList.append(['svm_rbf', tenFoldCV.mean(), svc_rbf_score, svc_rbf_best, tenFoldCV.std()])
print('3rd search, test acc = ' + str(svc_rbf_score))

#%% 3.2 SVM(Poly): Find Best SVC Hyperparameters by Telescope Search + Grid Search (Ploynomial Kernel)
# Kernels:
# ‘linear’, ‘poly’, ‘rbf’(default), ‘sigmoid’, ‘precomputed’ 
# poly: k = ( gamma*<x,x'>+coef0 )^degree
# http://scikit-learn.org/stable/modules/svm.html#kernel-functions
from sklearn.model_selection import StratifiedKFold, GridSearchCV
start_time = time.time()

coef0 = [1];    # Fix coef0
# 1st Search
svc_poly_degree_range_1 = 10. ** np.linspace(-3,8,12)
param_grid = dict(coef0=coef0, degree=svc_poly_degree_range_1)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_poly_best_para_1 = grid.best_params_
print('1st search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 2nd Search
svc_poly_degree_range_2 = np.linspace(svc_poly_best_para_1['degree']*0.1,svc_poly_best_para_1['degree']*10,10)
param_grid = dict(coef0=coef0, degree=svc_poly_degree_range_2)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_poly_best_para_2 = grid.best_params_
print('2nd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 3rd Search(save time)
"""
svc_poly_degree_range_3 = np.linspace(svc_poly_best_para_2['degree']*0.5,svc_poly_best_para_2['degree']*1.5,10)
param_grid = dict(coef0=coef0, degree=svc_poly_degree_range_3)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
"""
svc_poly_best_para_3 = grid.best_params_

# Choose best model/parameter
svc_poly_best = grid.best_estimator_
svc_poly_score = svc_poly_best.score(usedTestX,usedTestY)
tenFoldCV = cross_val_score(svc_poly_best, usedTrainX, usedTrainY, cv=10, n_jobs=-1)
#avgCV = np.mean(tenFoldCV)
usedModelList.append(['svm_poly',  tenFoldCV.mean(), svc_poly_score, svc_poly_best,  tenFoldCV.std()])
print('3rd search, test acc = ' + str(svc_poly_score))
print("Time(s):\t", (time.time() - start_time))

# 1st search, test acc = 0.6933333333333334
# 2nd search, test acc = 0.6933333333333334
# 3rd search, test acc = 0.6833333333333333
# Time(s):         252.0660674571991
'''
#%% 4. Model Comparsion, using Accuracy
from plotAcc import plotAcc
plotAcc(usedModelList, '')

#%% 5. Model Comparsion, using sign-test
"""
# [optional - Model Evaluation] 
    Compare at 2 differnet methods using 10 re-runs of a 10-fold cross-validation 
    and perform a suitable statistical test 
    to assess whether one of those performs significantly better than the others.
"""
# sklearn.model_selection.ShuffleSplit: Random permutation cross-validator
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
# http://scikit-learn.org/stable/modules/cross_validation.html#computing-cross-validated-metrics
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
