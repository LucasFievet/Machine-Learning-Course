import numpy as np

from sklearn.linear_model import Ridge 

from .cluster_window import get_cluster_mean
from .load_data import load_targets

def ridge_predict():
    y = load_targets()["Y"].tolist()
    X = get_cluster_mean(w_size=5,thresh=0.4)
    weights = [1.0/y.count(y[i]) for i in range(len(y))]
    clf = Ridge(alpha=1.0)
    clf.fit(X,y,weights)
    print("coefficient of determination R^2:",clf.score(X,y,weights))
