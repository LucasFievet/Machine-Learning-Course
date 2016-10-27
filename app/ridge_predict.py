import os
import pandas as pd
import numpy as np
from math import sqrt

from sklearn.linear_model import Ridge, Lasso 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn import preprocessing

from .cluster_window import get_count_gray
from .load_data import load_targets
from .settings import CURRENT_DIRECTORY
from .squared_error import squared_error

def ridge_predict():
    y = load_targets()["Y"].tolist()
    Xt = get_count_gray(w_size=3,thresh=0.10,training=True)
    Xp = get_count_gray(w_size=3,thresh=0.10,training=False)
    scaler = preprocessing.StandardScaler().fit(Xt)

    Xt = scaler.transform(Xt)
    Xp = scaler.transform(Xp)
    
    #weights = [sqrt(1.0/y.count(y[i])) for i in range(len(y))]
    weights = ([1.0/y.count(y[i]) for i in range(len(y))])
    #weights = [1.0 for i in range(len(y))]

    clf = Lasso(alpha=0.3,max_iter=10000)
    clf.fit(Xt,y,weights)
    predicted = cross_val_predict(clf, Xt, y, cv=5)
    print("Squared Error:",squared_error(y,predicted))
    # results = list(map(lambda x,y: (x+y)/2, results1,results2))
    result = {"ID": list(range(1,len(Xp)+1)), "Prediction": clf.predict(Xp)}
    result_path = os.path.join(CURRENT_DIRECTORY,"..","result.csv")
    pd.DataFrame(result).to_csv(result_path,index=False)

