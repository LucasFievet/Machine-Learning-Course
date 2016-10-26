import os
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.linear_model import Ridge, Lasso 
from sklearn import preprocessing
import sklearn.metrics as skmet
from sklearn.metrics import mean_squared_error
from sklearn.grid_search import GridSearchCV

from .cluster_window import get_cluster_mean
from .load_data import load_targets
from .settings import CURRENT_DIRECTORY

def ridge_predict():
    y = load_targets()["Y"].tolist()
    Xt = get_cluster_mean(w_size=5,thresh=0.4,training=True)
    Xp = get_cluster_mean(w_size=5,thresh=0.4,training=False)
    scaler = preprocessing.StandardScaler().fit(Xt)

    Xt = scaler.transform(Xt)
    Xp = scaler.transform(Xp)
    
    #weights = [sqrt(1.0/y.count(y[i])) for i in range(len(y))]
    weights = ([1.0/y.count(y[i]) for i in range(len(y))])
    #weights = [1.0 for i in range(len(y))]

    clf = Lasso(alpha=0.3,max_iter=10000)
    clf.fit(Xt,y,weights)
    results = clf.predict(Xp)
    #param = {'alpha':np.arange(0.0,10.0,0.1), 'fit_intercept':[True,False]}
    #neg_scorefun = skmet.make_scorer(lambda x, y: -RMSEscore(x, y))
    #gs = GridSearchCV(clf, param, scoring=neg_scorefun, cv=10)
    #gs.fit(Xt,y)
    #best_fit = gs.best_estimator_
    #print(best_fit)
    #print('best score =', -gs.best_score_)
    #results = best_fit.predict(Xp)
    #results = clf.predict(Xp)
    #print("coefficient of determination R^2:",clf1.score(Xt,y,weights))
    #clf2 = Ridge(alpha=1.0,max_iter=100000,normalize=False)
    #clf2.fit(Xt,y,weights)
    #print("coefficient of determination R^2:",clf2.score(Xt,y,weights))
    #results2 = clf2.predict(Xp)
    
    # results = list(map(lambda x,y: (x+y)/2, results1,results2))
    result = {"ID": list(range(1,len(Xp)+1)), "Prediction": results}
    result_path = os.path.join(CURRENT_DIRECTORY,"..","result.csv")
    pd.DataFrame(result).to_csv(result_path,index=False)

def RMSEscore(gtruth, pred):
    return mean_squared_error(gtruth, pred)
