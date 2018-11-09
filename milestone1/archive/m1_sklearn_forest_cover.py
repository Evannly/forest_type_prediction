#%% 0. Fundamental Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import os
#os.chdir('E:\xdocuments\Courses\2018_Spring\CSE517A_Machine_Learning\applications\milestone1')

#%% 1. Read dataset
# Read data
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
# Take a qucik look
# labeledData.head()
allX = labeledData.drop(['Id', 'Cover_Type'],axis=1)
allY = labeledData['Cover_Type']

'''
See number of samples of each category
allY.value_counts()
Out[101]: 
5    1647
6    1631
3    1630
7    1628
1    1608
2    1599
4    1597
'''

discreteCols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]
continuousCols = [x for x in allX.columns if x not in discreteCols]

#%% Vilization
# aa = allX['Elevation']
# plt.pyplot.hist(aa, bins=100)
# Relationship between variance and effect

#%% 2. Data preprocess
from sklearn.feature_selection import VarianceThreshold

# 2.1 Quantify(Dummy var)
# allX = pd.get_dummies(allX, dummy_na=True)

bool_cols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]

# 2.1 Normalization: centering and unification(??), for only continuous data
# numeric_feats = allX.dtypes[allX.dtypes != "object"].index
allX[continuousCols] = allX[continuousCols].apply(lambda x: (x - x.mean()) / (x.std()))
# allX[continuousCols] = allX[continuousCols].apply(lambda x: (x - x.mean()) / (x.mean()))
# allX[ContinuousCols] = allX[ContinuousCols].apply(lambda x: 1 / (1+np.exp(x)))
allX = allX.fillna(0)

'''
# 2.3 Remove features with low  variance
allX_columns = allX.columns
selector = VarianceThreshold(threshold=(0.05))
allXDimDeArray = selector.fit_transform(allX)
# labels = [allX_columns[x] for x in selector.get_support(indices=True) if x]
allX = pd.DataFrame(allXDimDeArray)
'''

# 2.3 Split
from sklearn.cross_validation import train_test_split
trainX, testX, trainY, testY = train_test_split(allX, allY, random_state=1)

# 2.3 Fill missing data (by taking average)
trainX = trainX.fillna(trainX.mean())
testX  =  testX.fillna(testX.mean())

# 2.4 Add Bias Dimension
allXCount = len(allX[allX.columns[0]])
allX['Bias'] = pd.Series(np.ones(allXCount), index=allX.index)

testXDis = testX[discreteCols];     testXCon = testX[continuousCols]
trainXDis = trainX[discreteCols];   trainXCon = trainX[continuousCols]


#%% 3. Experiment with regularizer coefficients
import sklearn.linear_model as lm
maxIter = 1000
tolerance = 1e-3
alp_count = 21
alp_accuracy = np.zeros([1,alp_count])
alp_list = np.linspace(0,20,alp_count);  alp_list[0]=0.001
error_record = np.zeros(1,alp_count)
for alp in alp_list:
    ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=alp)
    ridge.fit(trainX, trainY)
    print('ALPHA: ', alp, 'ACC: ', ridge.score(testX, testY))
    # Cross Validation
    # tenFoldCV = cross_val_score(model, trainX, trainY, cv=10)
    # print('average of 10VC: ', np.mean(tenFoldCV))
    
#%% 4. Feature study
# from sklearn.svm import SVC;        svm = SVC(kernel='linear')
from sklearn.svm import LinearSVC;  svmLinear = LinearSVC()
ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=0.5)
svmLinear.fit(trainX, trainY)
ridge.fit(trainX, trainY)

# Testing
ridge_score = ridge.score(testX, testY)
svm_score = svmLinear.score(testX, testY)
print('Ridge Testing Score:\t', round(ridge_score,4), '%')
print('L_SVM Testing Score:\t', round(svm_score,4), '%')

# Coefficient
ridge_coef = ridge.coef_
svm_coef = svmLinear.coef_
coef_count = ridge_coef.shape[1]
classifierCount = ridge_coef.shape[0]
plt.bar(np.arange(coef_count), np.sum(np.abs(svm_coef),0))
# plt.legend(['Linear SVM', 'ridge'])
plt.bar(np.arange(coef_count), np.sum(np.abs(ridge_coef),0))
# plt.legend(['ridge'])
plt.legend(['Linear SVM', 'ridge'])
plt.xlabel('weights')
plt.ylabel('abs(weight)')

svm_coef_abs = np.sum(np.abs(svm_coef),0)
ridge_coef_abs = np.sum(np.abs(ridge_coef),0)
opacity = 0.40
# plt.figure(figsize=(10,6))
plt.bar(np.arange(coef_count), svm_coef_abs/np.linalg.norm(svm_coef_abs), alpha=opacity)
plt.bar(np.arange(coef_count), ridge_coef_abs/np.linalg.norm(ridge_coef_abs),  alpha=opacity)
# plt.legend(['Linear SVM', 'ridge'])
# plt.bar(np.arange(coef_count), np.sum(np.abs(ridge_coef),0))
# plt.legend(['ridge'])
plt.legend(['Linear SVM', 'ridge'])
label_list=['Elevation',
'Aspect',
'Slope',
'HoriD_Hydr',
'VertD_Hydr',
'HoriD_Road',
'HShade_9am',
'HShade_Noon',
'HShade_3pm',
'HoriD_FirePoints']
# 'Wild_Area',
# 'Soil_Type']
for x in range(4): label_list.append('Wild_Area'+str(x+1))
for x in range(40): label_list.append('Soil_Type'+str(x+1))
plt.xticks(np.arange(54),label_list,size='small',rotation=80)
plt.xlabel('weights')
plt.ylabel('abs(weight)')

# plt.show()
# xDrawBar3D(coef_count, classifierCount, np.abs(ridge_coef))
# xDrawBar3D(coef_count, classifierCount, np.abs(svm_coef))

def xDrawBar3D(coefCount, classifierCount, coefMat):
    # import numpy as np
    # import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
        
    # setup the figure and axes
    fig = plt.figure()
    # fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    
    # fake data
    _x = np.arange(coefCount)
    _y = np.arange(classifierCount)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    
    # top = x + y
    top = coefMat.flatten()
    bottom = np.zeros_like(top)
    width = depth = 1
    
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True)    
    
    plt.show()


#%% 3D Bar Test
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# fake data
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
ax2.set_title('Not Shaded')

plt.show()

#%% 5. Different CLassifiers
# Choose model
# from sklearn import gaussian_process
# Gaussian = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
# GaussianProcessRegressor
# from sklearn import metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC;        svm = SVC(kernel='linear')
from sklearn.svm import LinearSVC;  svmLinear = LinearSVC()
from sklearn import tree;           cartTree = tree.DecisionTreeClassifier()
'''
class sklearn.linear_model.SGDClassifier(
        loss=’hinge’, 
        # possible options are ‘hinge’, ‘log’, ‘modified_huber’, 
        # ‘squared_hinge’, ‘perceptron’, 
        # or a regression loss: ‘squared_loss’, ‘huber’, 
        # ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
        penalty=’l2’, alpha=0.0001, # Defaults to 0.0001
        l1_ratio=0.15, 
        fit_intercept=True, 
        max_iter=None, 
        tol=None, 
        shuffle=True, 
        verbose=0, 
        epsilon=0.1, n_jobs=1, 
        random_state=None, learning_rate=’optimal’, eta0=0.0, 
        power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)
'''

linear_square = lm.SGDClassifier(loss='squared_loss', penalty='none', max_iter=maxIter, tol=tolerance)
ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=0.5)
# ridgel1 = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance)
lasso = lm.SGDClassifier(loss='squared_loss', penalty='l1', max_iter=maxIter, tol=tolerance)
logisitc = lm.LogisticRegression()
# bayes = lm.BayesianRidge()


# Bagging
'''
class sklearn.ensemble.BaggingClassifier(
        base_estimator=None, n_estimators=10, 
        max_samples=1.0, max_features=1.0, 
        bootstrap=True, bootstrap_features=False, oob_score=False, 
        warm_start=False, n_jobs=1, random_state=None, verbose=0)
'''
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
        ['CART Decision Tree', cartTree],        
        ['CART bagging', cartTree_bagging],
        ['CART Adaboost', cartTree_Adaboost],
        ['Logistic', logisitc],
        ['Logistic Bagging', logisitc_bagging]
        ]

import time
from sklearn.model_selection import cross_val_score
# usedModelList = np.zeros(len(modelList))
usedModelList = []
for name, model in modelList:
    start_time = time.time()
    # Training / Fitting
    print('Name: ',name)
    model.fit(trainXDis, trainY)
    
    # Cross Validation
    tenFoldCV = cross_val_score(model, trainXDis, trainY, cv=10)
    avgCV = np.mean(tenFoldCV)
    print('average of 10VC: ', avgCV)
    
    # Testing
    score = model.score(testXDis, testY)
    print('Testing Score:\t', round(score,2), '%')
    # print('Testing Score: ', score)
    
    # Running Time
    print("Time(s):\t", (time.time() - start_time))
    
    usedModelList.append([name, avgCV, score])
    
    # print('\n')
xDrawPillar(usedModelList, 'CV and Test result, discrete features')
    
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
    rects1 = plt.bar(index, CVs, bar_width,alpha=opacity, color='b',label=    '10-fold Cross Validation')    
    rects2 = plt.bar(index + bar_width, scores, bar_width,alpha=opacity,color='r',label='Test Accuracy')    
         
    plt.xlabel('Models')    
    plt.ylabel('Scores')    
    plt.title(title)
    plt.xticks(index + bar_width -0.15, names)
    # plt.ylim(0,40);    
    plt.legend();    
      
    plt.tight_layout();   
    plt.show();       