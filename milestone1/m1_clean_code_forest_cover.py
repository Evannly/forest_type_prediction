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
It's quite balanced!
'''

discreteCols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]
continuousCols = [x for x in allX.columns if x not in discreteCols]

allXDeUnit = allX.apply(lambda x: (x) / (x.mean()))
featurevars = allXDeUnit.var().fillna(0)
# featurevars
plt.hist(np.array(featurevars), bins=55)

#%% 2. Visualization

#%% 3. Data preprocess
from sklearn.feature_selection import VarianceThreshold

# 2.1 Quantify(Encoding categorical features into dummy variable)
# allX = pd.get_dummies(allX, dummy_na=True) 

bool_cols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]

# 2.1 Normalization: centering and unification(??), for only continuous data
allX[continuousCols] = allX[continuousCols].apply(lambda x: (x - x.mean()) / (x.std()))
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

# 2.4 Fill missing data (by taking average)
trainX = trainX.fillna(trainX.mean())
testX  =  testX.fillna(testX.mean())

# 2.4 Add Bias Dimension
allXCount = len(allX[allX.columns[0]])
allX['Bias'] = pd.Series(np.ones(allXCount), index=allX.index)

testXDis = testX[discreteCols];     testXCon = testX[continuousCols]
trainXDis = trainX[discreteCols];   trainXCon = trainX[continuousCols]

#%% 4. Experiment with regularizer coefficients
from sklearn.model_selection import cross_val_score
import sklearn.linear_model as lm
maxIter = 1000
tolerance = 1e-3
alp_count = 21
alp_accuracy = np.zeros([1,alp_count])
alp_list = np.linspace(0,5,alp_count);  alp_list[0]=0.001
scoreRecord = np.zeros(alp_count)
avgCVRecord = np.zeros(alp_count);
idx = 0;
for alp in alp_list:
    ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2',     max_iter=maxIter, tol=tolerance, alpha=alp)
    ridge.fit(trainX, trainY)
    score = ridge.score(testX, testY)
    # Cross Validation
    tenFoldCV = cross_val_score(ridge, trainX, trainY, cv=10)
    avgCV = np.mean(tenFoldCV)
    avgCVRecord[idx] = avgCV
    scoreRecord[idx] = score
    idx += 1    
    print('ALPHA: ', alp, 'ACC: ', score)
    print('average of 10VC: ', avgCV)
plt.plot(avgCVRecord)
plt.plot(scoreRecord)
plt.legend(['10-fold CV','test score'])
# plt.xticks(np.arange(54),label_list,size='small',rotation=80)
plt.xlabel('regularizer')
plt.ylabel('Accuracy')
plt.xticks(np.arange(21),alp_list)
    
#%% 5. Feature study
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
svm_coef_abs = np.sum(np.abs(svm_coef),0)
ridge_coef_abs = np.sum(np.abs(ridge_coef),0)
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
for x in range(4): label_list.append('Wild_Area'+str(x+1))
for x in range(40): label_list.append('Soil_Type'+str(x+1))

plt.figure()
opacity = 0.40
plt.bar(np.arange(coef_count), svm_coef_abs/np.linalg.norm(svm_coef_abs), alpha=opacity)
plt.bar(np.arange(coef_count), ridge_coef_abs/np.linalg.norm(ridge_coef_abs),  alpha=opacity)
plt.legend(['Linear SVM', 'ridge'])

plt.xticks(np.arange(54),label_list,size='small',rotation=80)
plt.xlabel('weights')
plt.ylabel('abs(weight)')

#%% 6. Try Different CLassifiers
# Import models
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC;        svm = SVC(kernel='linear')
from sklearn.svm import LinearSVC;  svmLinear = LinearSVC()
from sklearn import tree;           cartTree = tree.DecisionTreeClassifier()

linear_square = lm.SGDClassifier(loss='squared_loss', penalty='none', max_iter=maxIter, tol=tolerance)
ridge = lm.SGDClassifier(loss='squared_loss', penalty='l2', max_iter=maxIter, tol=tolerance, alpha=0.5)
lasso = lm.SGDClassifier(loss='squared_loss', penalty='l1', max_iter=maxIter, tol=tolerance)
logisitc = lm.LogisticRegression()


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
        # ['Logistic Bagging', logisitc_bagging]
        ['CART Decision Tree', cartTree],        
        ['CART bagging', cartTree_bagging],
        ['CART Adaboost', cartTree_Adaboost]             
        ]

import time
from sklearn.model_selection import cross_val_score
usedModelList = []
usedTrainX = trainXCon
usedTestX = testXCon
usedTitle = 'continuous features'

for name, model in modelList:
    start_time = time.time()
    # Training / Fitting
    print('Name: ',name)
    model.fit(usedTrainX, trainY)
    
    # Cross Validation
    tenFoldCV = cross_val_score(model, usedTrainX, trainY, cv=10)
    avgCV = np.mean(tenFoldCV)
    print('average of 10VC: ', avgCV)
    
    # Testing
    score = model.score(usedTestX, testY)
    print('Testing Score:\t', round(score,2)*100, '%')
    # print('Testing Score: ', score)
    
    # Running Time
    print("Time(s):\t", (time.time() - start_time))
    
    usedModelList.append([name, avgCV, score])
    
    # print('\n')
xDrawPillar(usedModelList, 'CV and Test result, ' + usedTitle)
    
# def xVote(modelList):
def xDrawPillar(usedModelList, title):    
    n_groups = len(usedModelList);
    names = [item[0] for item in usedModelList]
    CVs   = [item[1] for item in usedModelList]
    scores= [item[2] for item in usedModelList]
         
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
         
    opacity = 0.4    
    rects1 = plt.bar(index, CVs, bar_width,alpha=opacity, color='b',label=    '10-fold Cross Validation')    
    rects2 = plt.bar(index + bar_width, scores, bar_width,alpha=opacity,color='r',label='Test Accuracy')    
         
    plt.xlabel('Models')    
    plt.ylabel('Scores')    
    plt.title(title)
    plt.xticks(index + bar_width -0.15, names)    
    plt.legend();
      
    plt.tight_layout();   
    plt.show();       
    plt.ylim(ymax=1)