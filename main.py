"""Main file."""

import sys
import os
from time import strftime

from app.heatmap import heatmap
from app.locations_3Dplot import locations_3D
from app.predictor_cluster import predict_cluster
from app.heatmap_side import heatmap_side
from app.submission import submission_predictor
from app.ridge_predict import ridge_predict
from app.simple_ratio import simple_ratio

from app.settings import CACHE_DIRECTORY, PLOT_DIRECTORY, ITERATE_DIRECTORY

if __name__ == "__main__":
    print(strftime("%Y-%m-%d %H:%M:%S"))
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    if not os.path.exists(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

    if not os.path.exists(ITERATE_DIRECTORY):
        os.makedirs(ITERATE_DIRECTORY)

    args = sys.argv
    if len(args) < 2:
        print("You need to specify a function")
    else:
        if args[1] == "heatmap":
            heatmap()
        elif args[1] == "heatmap_side":
            heatmap_side()
        elif args[1] == "locations_3D":
            locations_3D()
        elif args[1] == "scan_volume":
            submission_predictor()
        elif args[1] == "ridge":
            ridge_predict() 
        elif args[1] == "predict_cluster2":
            predict_cluster()
        elif args[1] == "ratio":
            simple_ratio()


