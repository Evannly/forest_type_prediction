# -*- coding: utf-8 -*-
'''
# Classifier Comparison
    > http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Multilabel classification
    > http://scikit-learn.org/stable/auto_examples/plot_multilabel.html

# [milestone 2 - Gaussian Process] Train and run Gaussian Processes.
    Evaluate and compare the predictions using at least two differnt kernels 
    via 10-fold cross-validation with a suitable error measure 
    (we recommend negative log predictiv density as it takes the predictive uncertainty into account).
    
    - Kernel types
    - Hyperparameter

# [optional - Support Vector Machine]
    Train and run a kernel Support Vector Machine.
    Evaluate the predictions using 10-fold cross-validation and a suitable error measure.
# [optional - Model Evaluation] 
    Compare at 2 differnet methods using 10 re-runs of a 10-fold cross-validation 
    and perform a suitable statistical test 
    to assess whether one of those performs significantly better than the others.
# [optional - Clustering]
    Cluster the features of your dataset (not using the labels). 
    Evaluate the clusters using the observed data/labels.
    - 0. KMeans, Mix Gaussian, Spectral
    - 1. Fix cluster number to 7, compute accuracy
    - 2. Learn cluster number from data by BIC
        > http://scikit-learn.org/stable/modules/mixture.html
    - Compare: Speed, accuracy
        > http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
'''

#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc
# Read data
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
#trainX, trainY, testX, testY = preproc(labeledData)
trainBatchSize = 1000
testBatchSize = 300
trainXBatch = trainX.iloc[0:trainBatchSize]
trainYBatch = trainY.iloc[0:trainBatchSize]
testXBatch = testX.iloc[0:testBatchSize]
testYBatch = testY.iloc[0:testBatchSize]

#%% 2. Gaussian Process
# class sklearn.gaussian_process.GaussianProcessRegressor(
#        kernel=None, alpha=1e-10, optimizer=’fmin_l_bfgs_b’, 
#        n_restarts_optimizer=0, max_iter_predict=100, 
#        warm_start=False, copy_X_train=True, 
#        random_state=None, multi_class=’one_vs_rest’, n_jobs=1)
# http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
# --- #
# Multiclass and multilabel algorithms
# http://scikit-learn.org/stable/modules/multiclass.html
# --- #
# Seleting hyper-parameter C and gamma of a RBF-Kernel SVM
# http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/svm/plot_svm_parameters_selection.html
# --- #
# Tuning the hyper-parameters of an estimator
# http://scikit-learn.org/stable/modules/grid_search.html
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, Exponentiation
from sklearn.svm import SVC

svm = SVC()
modelList = [
        ['SVM', svm]]

# Full kernel list:
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process
kernelList = [
        ['RBF',RBF()],
        ['Matern',Matern()],
        ['Exponentiation',Exponentiation()],
        ['Constant', ConstantKernel()]       
        ]

for name, gp_kernel in kernelList:
    gpc = GaussianProcessClassifier(kernel=gp_kernel,multi_class='one_vs_one',n_jobs=-1)
    modelList.append(['GP_'+name,gpc])


usedTrainX = trainXBatch
usedTrainY = trainYBatch
usedTestX = testXBatch
usedTestY = testYBatch
usedTitle = 'on small batch'
usedModelList = []
import time
from sklearn.model_selection import cross_val_score
for name, model in modelList:
    start_time = time.time()
    # Training / Fitting
    print('Name: ',name)
    model.fit(usedTrainX, usedTrainY)
    
    # Cross Validation
    #tenFoldCV = cross_val_score(gpc, usedTrainX, usedTrainY, cv=10)
    #avgCV = np.mean(tenFoldCV)
    #print('average of 10VC: ', avgCV)
    
    # Testing
    score = model.score(usedTestX, usedTestY)
    print('Testing Score:\t', round(score,2)*100, '%')
    # print('Testing Score: ', score)
    
    # Running Time
    print("Time(s):\t", (time.time() - start_time))
    
    usedModelList.append([name, 0, score, model])
    #usedModelList.append([name, avgCV, score, model])
    #usedModelList.append([name, score, model])

#%% Find Best SVC Hyperparameters by Telescope Search + Grid Search (RBF Kernel)
# http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/auto_examples/svm/plot_svm_parameters_selection.html
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    
from sklearn.model_selection import StratifiedKFold, GridSearchCV
start_time = time.time()

# Kernels:
# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
# 1st Search
svc_C_range_1 = 10. ** np.linspace(-3,8,12)
svc_gamma_range_1 = 10. ** np.linspace(-5,4,10)
param_grid = dict(gamma=svc_gamma_range_1, C=svc_C_range_1)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_best_para_1 = grid.best_params_
print('1st search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 2nd Search
svc_C_range_2 = np.linspace(svc_best_para_1['C']*0.1,svc_best_para_1['C']*10,10)
svc_gamma_range_2 = np.linspace(svc_best_para_1['gamma']*0.1,svc_best_para_1['gamma']*10,10)
param_grid = dict(gamma=svc_gamma_range_2, C=svc_C_range_2)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_best_para_2 = grid.best_params_
print('2nd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))

# 3rd Search
svc_C_range_3 = np.linspace(svc_best_para_2['C']*0.5,svc_best_para_2['C']*1.5,10)
svc_gamma_range_3 = np.linspace(svc_best_para_2['gamma']*0.5,svc_best_para_2['gamma']*1.5,10)
param_grid = dict(gamma=svc_gamma_range_3, C=svc_C_range_3)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_best_para_3 = grid.best_params_
print('3rd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))
print("Time(s):\t", (time.time() - start_time))
# score: 0.6833333333333333
# {'C': 733.3333333333334, 'gamma': 0.00575}
# 25 min

#%% Find Best SVC Hyperparameters by Telescope Search + Grid Search (Ploynomial Kernel)
# Kernels:
# ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 
# poly: k = ( gamma*<x,x'>+coef0 )^degree
# http://scikit-learn.org/stable/modules/svm.html#kernel-functions
from sklearn.model_selection import StratifiedKFold, GridSearchCV
start_time = time.time()

coef0 = [1];

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

# 3rd Search
svc_poly_degree_range_3 = np.linspace(svc_poly_best_para_2['degree']*0.5,svc_poly_best_para_2['degree']*1.5,10)
param_grid = dict(coef0=coef0, degree=svc_poly_degree_range_3)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid.fit(usedTrainX, usedTrainY)
svc_poly_best_para_3 = grid.best_params_
print('3rd search, test acc = ' + str(grid.best_estimator_.score(usedTestX,usedTestY)))
print("Time(s):\t", (time.time() - start_time))
# 1st search, test acc = 0.6933333333333334
# 2nd search, test acc = 0.6933333333333334
# 3rd search, test acc = 0.6833333333333333
# Time(s):         252.0660674571991
# {'C': 733.3333333333334, 'gamma': 0.00575}

#SVC3 = SVC(kernel='poly',degree=3,coef0=1)
#SVC3.fit(usedTrainX, usedTrainY)
#SVC3.score(usedTestX,usedTestY)

#%% Find Best Kernel Parameters
# hyperparameters are automatically optimized!
"""
# https://github.com/scikit-learn/scikit-learn/issues/6583
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, ExpSineSquared
from sklearn.model_selection import StratifiedKFold, GridSearchCV

ker_rbf = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")
ker_rq = ConstantKernel(1.0, constant_value_bounds="fixed") * RationalQuadratic(alpha=0.1, length_scale=1)
ker_expsine = ConstantKernel(1.0, constant_value_bounds="fixed") * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
kernel_list = [ker_rbf, ker_rq, ker_expsine]

#gpc = GaussianProcessClassifier(kernel=gp_kernel,multi_class='one_vs_one',n_jobs=-1)
gpk = GaussianProcessClassifier()
# gpk.get_params().keys()
param_grid = {'kernel': kernel_list,
              #"multi_class": ["one_vs_one"],
              'n_jobs': [-1]}
              #'alpha': [1e1]}
              #"optimizer": ["fmin_l_bfgs_b"]}
              #"n_restarts_optimizer": [1, 2, 3],
              #"normalize_y": [False],
              #"copy_X_train": [True], 
              #"random_state": [0]
              

grid_g_kernel = GridSearchCV(gpk, param_grid=param_grid, cv=StratifiedKFold(n_splits=5))
grid_g_kernel.fit(usedTrainX, usedTrainY)
"""

#%% 3. Plot accuracy
from plotAcc import plotAcc
plotAcc(usedModelList, 'CV and Test result, ' + usedTitle)
