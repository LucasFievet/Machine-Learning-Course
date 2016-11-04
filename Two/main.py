"""Main file."""

import sys
import os
import numpy as np
from time import strftime

from app.trial_one import submission_predictor
from app.healthy_sick_comparison import healthy_sick_comparison
from app.reduce_histogram import ReduceHistogram 
from app.find_features import FindFeatures

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
        if args[1] == "trial_one":
            submission_predictor()
        if args[1] == "healthy_sick_comparison":
            healthy_sick_comparison()
        if args[1] == "reduce_histogram":
            r = ReduceHistogram(10,20)
        if args[1] == "ff":
            f = FindFeatures()
            f.test()
