# Use SKLearn to do house price prediction
# REF: 
# (Chinese) https://morvanzhou.github.io/tutorials/machine-learning/sklearn/2-3-database/
# (使用sklearn进行数据挖掘-房价预测(1)) http://www.cnblogs.com/wxshi/p/7725814.html
# (Kaggle) https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# [More Linear Models of scikit-learn] http://scikit-learn.org/stable/modules/linear_model.html


#%% 0. LIBRARIES
#from mxnet import ndarray as nd
#from mxnet import autograd
#from mxnet import gluon
# from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
import sklearn.linear_model as lm
from sklearn import metrics
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# %matplotlib inline
import pandas as pd
import numpy as np


#%% 1. Read dataset
# Read data
train = pd.read_csv("../data/kaggle_house_pred_train.csv")
test = pd.read_csv("../data/kaggle_house_pred_test.csv")
all_X = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))
# Check data fields(features) and first 5 data, Visualization
#train.head()
#train.info()
#train.hist(bins=50,figsize=(15,10))#bins 柱子个数
#plt.show()
#plt.savefig('a.jpg')  #保存图片

#%% 2. Data preprocess
# 2.1 Normalization: centering and unification
numeric_feats = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feats] = all_X[numeric_feats].apply(lambda x: (x - x.mean())
                                                            / (x.std()))

# 2.2 Quantify(Dummy var)
all_X = pd.get_dummies(all_X, dummy_na=True)

# 2.3 Fill missing data (by taking average)
all_X = all_X.fillna(all_X.mean())

# 2.4 Split data into Training and Testing Set
num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.SalePrice.as_matrix()

# 2.5 Transform into NDArray to use GLUON
#X_train = nd.array(X_train)
#y_train = nd.array(y_train)
#y_train.reshape((num_train, 1))
#
#X_test = nd.array(X_test)

from sklearn.cross_validation import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X_train, y_train, random_state=1)

#%% 3. Learning and Testing
# Choose model
from sklearn import gaussian_process
Gaussian = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
from sklearn.svm import SVR
svm = SVR(C=1.0, epsilon=0.2)
from sklearn import tree
cartTree = tree.DecisionTreeRegressor()

linear_square = lm.LinearRegression()
ridge = lm.Ridge(alpha=0.5)
lasso = lm.Lasso(alpha=0.1)
bayes = lm.BayesianRidge()


modelList = [
        ['linear_square',linear_square], 
        ['ridge',ridge], 
        ['lasso',lasso], 
        ['Bayes',bayes], 
        ['Gaussian Proceee', Gaussian], 
        ['SVM', svm], 
        ['CART Decision Tree', cartTree]]


for name, model in modelList:
    # Training / Fitting
    model.fit(Xtr, ytr)
    print('Name: ',name)
    
    # Cross Validation
    # print('10VC: ', cross_val_score(model, Xtr, ytr, cv=10))
    
    # Testing
    yte_pred = model.predict(Xte)
    print('Testing Score: ', model.score(Xte, yte))
    kaggleScore = np.sqrt(2*metrics.mean_squared_error(np.log(yte), np.log(np.abs(yte_pred))))
    print('Kaggle Score: ', kaggleScore)
    
    print('\n\n')
    


#%% Visualization
