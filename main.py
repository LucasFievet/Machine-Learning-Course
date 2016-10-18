"""Main file."""


import sys
import os


from app.load_data import load_sample_input
from app.heatmap import heatmap
from app.predictor import predict
from app.predictor_cut import predict_cut
from app.heatmap_side import heatmap_side

from app.settings import CACHE_DIRECTORY, PLOT_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "12/10/2016"
__status__ = "Production"


if __name__ == "__main__":
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    if not os.path.exists(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

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

