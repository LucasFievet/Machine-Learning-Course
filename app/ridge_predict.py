import os
import pandas as pd
from sklearn.linear_model import Ridge 

from .cluster_window import get_cluster_mean
from .load_data import load_targets
from .settings import CURRENT_DIRECTORY

def ridge_predict():
    y = load_targets()["Y"].tolist()
    Xt = get_cluster_mean(w_size=5,thresh=0.4,training=True)
    weights = [1.0/y.count(y[i]) for i in range(len(y))]
    clf = Ridge(alpha=1.0)
    clf.fit(Xt,y,weights)
    print("coefficient of determination R^2:",clf.score(Xt,y,weights))

    Xp = get_cluster_mean(w_size=5,thresh=0.4,training=False)
    result = {"ID": list(range(1,len(Xp)+1)), "Predicted": clf.predict(Xp)}
    result_path = os.path.join(CURRENT_DIRECTORY,"..","result.csv")
    pd.DataFrame(result).to_csv(result_path,index=False)


