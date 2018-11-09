# Data preprocessing for forest cover type prediction
def preproc(labeledData, norm='norm', rmLowVar='NOTrmLowVar', split='spilt', bias='bias'):
    import pandas as pd
    import numpy as np
    allX = labeledData.drop(['Id', 'Cover_Type'],axis=1)
    allY = labeledData['Cover_Type']
    discreteCols = [col for col in allX 
             if allX[[col]].dropna().isin([0, 1]).all().values]
    continuousCols = [x for x in allX.columns if x not in discreteCols]
    
    #allXDeUnit = allX.apply(lambda x: (x) / (x.mean()))
    #featurevars = allXDeUnit.var().fillna(0)

    
    
    # 2.1 Quantify(Encoding categorical features into dummy variable)
    # allX = pd.get_dummies(allX, dummy_na=True) 
    
    bool_cols = [col for col in allX 
                 if allX[[col]].dropna().isin([0, 1]).all().values]
    
    # 2.1 Normalization: centering and unification(??), for only continuous data
    allX[continuousCols] = allX[continuousCols].apply(lambda x: (x - x.mean()) / (x.std()))
    allX = allX.fillna(0)
    
    '''
    # 2.3 Remove features with low  variance
    #from sklearn.feature_selection import VarianceThreshold
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
    
    return trainX, trainY, testX, testY, trainXDis, testXDis, trainXCon, testXCon

# Decide used data: data used in following stages
def setUsedData(dataMode, trainX, trainY, testX, testY, trainBatchSize=1000, testBatchSize=300):
    # Take batch to speed up
    trainXBatch = trainX.iloc[0:trainBatchSize]
    trainYBatch = trainY.iloc[0:trainBatchSize]
    testXBatch = testX.iloc[0:testBatchSize]
    testYBatch = testY.iloc[0:testBatchSize]
#    trainXBatch = trainX.iloc[0:trainBatchSize]
#    trainYBatch = trainY.iloc[0:trainBatchSize]
#    testXBatch = testX.iloc[0:testBatchSize]
#    testYBatch = testY.iloc[0:testBatchSize]
    if dataMode == 'batch':
        usedTrainX = trainXBatch
        usedTrainY = trainYBatch
        usedTestX = testXBatch
        usedTestY = testYBatch
        usedTitle = 'on small batch'    
    elif dataMode == 'all':
        usedTrainX = trainX
        usedTrainY = trainY
        usedTestX = testX
        usedTestY = testY
        usedTitle = 'on full data'
    else:
        print('Unknown Data Mode')
        return
    return usedTrainX, usedTrainY, usedTestX, usedTestY, usedTitle