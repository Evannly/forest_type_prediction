"""
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np
def xTeleGridSearch(model, initParas, iterCount, X, Y, n_splits=5):
    for iter = range(iterCount):
        svc_C_range = 10. ** np.linspace(-3,8,12)
        svc_gamma_range = 10. ** np.linspace(-5,4,10)
        param_grid = dict(gamma=svc_gamma_range_1, C=svc_C_range_1)
        grid = GridSearchCV(model, param_grid=param_grid, cv=StratifiedKFold(n_splits))
        grid.fit(X, Y)
        svc_best_para = grid.best_params_
    
   """ 