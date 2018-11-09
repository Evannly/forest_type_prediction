# Use SKLearn to do house price prediction
# REF: 
# (Chinese) https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-3-database/
# (使用sklearn进行数据挖掘-房价预测(1)) http://www.cnblogs.com/wxshi/p/7725814.html
# (Kaggle) https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# [More Linear Models of scikit-learn] http://scikit-learn.org/stable/modules/linear_model.html


#%% 0. LIBRARIES
import sklearn.linear_model as lm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np


#%% 1. Read dataset
# Read data
labeledData = pd.read_csv("../data/kaggle_house_pred_train.csv")
allX = labeledData.drop(['Id', 'SalePrice'],axis=1);
allY = labeledData['SalePrice'];

#%% 2. Data preprocess
# 2.1 Normalization: centering and unification
numeric_feats = allX.dtypes[allX.dtypes != "object"].index
allX[numeric_feats] = allX[numeric_feats].apply(lambda x: (x - x.mean()) / (x.std()))

# 2.2 Quantify(Dummy var)
allX = pd.get_dummies(allX, dummy_na=True)

# 2.3 Split
trainX, testX, trainY, testY = train_test_split(allX, allY, random_state=1)

# 2.3 Fill missing data (by taking average)
trainX = trainX.fillna(trainX.mean())
testX  =  testX.fillna(testX.mean())

#%% 3. Learning and Testing
# Choose model
from sklearn import gaussian_process
from sklearn.svm import SVR
from sklearn import tree

linear_square = lm.LinearRegression()
ridge = lm.Ridge(alpha=0.5)
lasso = lm.Lasso(alpha=0.1)
bayes = lm.BayesianRidge()
Gaussian = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
cartTree = tree.DecisionTreeRegressor()
svm = SVR(C=1.0, epsilon=0.2)

modelList = [
        #['linear_square',linear_square], 
        ['ridge',ridge], 
        ['lasso',lasso], 
        ['Bayes',bayes], 
        ['Gaussian Process', Gaussian], 
        ['SVM', svm], 
        ['CART Decision Tree', cartTree]]


for name, model in modelList:
    # Training / Fitting
    model.fit(trainX, trainY)
    print('Name: ',name)
    
    # Cross Validation
    # print('10VC: ', cross_val_score(model, Xtr, ytr, cv=10))
    
    # Testing
    testY_pred = model.predict(testX)
    print('Testing Score: ', model.score(testX, testY))
    kaggleScore = np.sqrt(2*metrics.mean_squared_error(np.log(testY), np.log(testY_pred)))
    print('Kaggle Score: ', kaggleScore)
    
    print('\n\n')
    


#%% Visualization
