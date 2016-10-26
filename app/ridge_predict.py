import os
import pandas as pd
from math import sqrt
from sklearn.linear_model import Ridge, Lasso 

from .cluster_window import get_cluster_mean
from .load_data import load_targets
from .settings import CURRENT_DIRECTORY

def ridge_predict():
    y = load_targets()["Y"].tolist()
    Xt = get_cluster_mean(w_size=5,thresh=0.4,training=True)
    Xp = get_cluster_mean(w_size=5,thresh=0.4,training=False)
    weights = [sqrt(1.0/y.count(y[i])) for i in range(len(y))]
    clf1 = Lasso(alpha=100.0,max_iter=100000,normalize=False)
    clf1.fit(Xt,y,weights)
    print("coefficient of determination R^2:",clf1.score(Xt,y,weights))

    results1 = clf1.predict(Xp)
    clf2 = Ridge(alpha=100.0,max_iter=100000,normalize=False)
    clf2.fit(Xt,y,weights)
    print("coefficient of determination R^2:",clf2.score(Xt,y,weights))
    results2 = clf2.predict(Xp)

    results = list(map(lambda x,y: (x+y)/2, results1,results2))
    result = {"ID": list(range(1,len(Xp)+1)), "Prediction": results}
    result_path = os.path.join(CURRENT_DIRECTORY,"..","result.csv")
    pd.DataFrame(result).to_csv(result_path,index=False)


