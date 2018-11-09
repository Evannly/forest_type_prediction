#%% 1. Data reading and preprocessing
import pandas as pd
from preprocessing import preproc
# Read data
labeledData = pd.read_csv("../data/kaggle_forest_cover_train.csv")
trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon = preproc(labeledData)
#trainX, trainY, testX, testY = preproc(labeledData)
trainBatchSize = 300
testBatchSize = 100
trainXBatch = trainX.iloc[0:trainBatchSize]
trainYBatch = trainY.iloc[0:trainBatchSize]
testXBatch = testX.iloc[0:testBatchSize]
testYBatch = testY.iloc[0:testBatchSize]

#%% 2. Define Model
import GPy
import numpy as np
# from matplotlib import pyplot as plt
k = GPy.kern.RBF(1, variance=10., lengthscale=0.1)
lik = GPy.likelihoods.Gaussian()
#m = GPy.core.GP(X=trainX,
Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
#m = GPy.core.sparse_gp.SparseGP(X=trainX,
m = GPy.core.GP(X=trainXBatch,
                Y=trainYBatch.values.reshape(300,1), 
                kernel=k, 
                #Z=Z,
                inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),                
                likelihood=lik)

print(m, '\n')
for i in range(50):
    m.optimize('bfgs', max_iters=100) #first runs EP and then optimizes the kernel parameters
    print('iteration:', i)
    print(m)
    print("")
    
m = GPy.models.GPClassification(trainXBatch,trainYBatch.values.reshape(300,1))