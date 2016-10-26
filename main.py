"""Main file."""

import sys
import os

from app.load_data import load_sample_input
from app.heatmap import heatmap
from app.predictor import predict
from app.predictor_cluster import predict_cluster
from app.predictor_cut import predict_cut
from app.predictor_cut_iterate import predict_cut_iterate
from app.heatmap_side import heatmap_side
from app.experimental import scan_volume
from app.cluster_window import get_clusters, get_cluster_mean
from app.ridge_predict import ridge_predict

from app.settings import CACHE_DIRECTORY, PLOT_DIRECTORY, ITERATE_DIRECTORY

if __name__ == "__main__":
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
        if args[1] == "load":
            print(load_sample_input())
        elif args[1] == "heatmap":
            heatmap()
        elif args[1] == "predict":
            predict()
        elif args[1] == "heatmap_side":
            heatmap_side()
        elif args[1] == "predict_cut":
            predict_cut()
        elif args[1] == "scan_volume":
            scan_volume()
        elif args[1] == "predict_cluster":
            get_clusters(w_size=5,thresh=0.4)
        elif args[1] == "ridge":
            ridge_predict() 
        elif args[1] == "predict_cluster2":
            predict_cluster()
        elif args[1] == "predict_cut_iterate":
            if len(args) < 3:
                print("Additional Argument needed for this!")
            else:
                predict_cut_iterate(num=int(args[2]))

