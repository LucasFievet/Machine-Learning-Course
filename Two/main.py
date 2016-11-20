"""Main file."""

import sys
import os
import time
import numpy as np

from app.trial_one import submission_predictor
from app.healthy_sick_comparison import healthy_sick_comparison
from app.reduce_histogram import ReduceHistogram 
from app.find_features import FindFeatures
from app.predictor import predict

from app.settings import CACHE_DIRECTORY, PLOT_DIRECTORY

if __name__ == "__main__":
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    if not os.path.exists(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

    args = sys.argv
    if len(args) < 2:
        print("You need to specify a function")
    else:
        if args[1] == "trial_one":
            submission_predictor()
        if args[1] == "healthy_sick_comparison":
            healthy_sick_comparison()
        if args[1] == "reduce_histogram":
            r = ReduceHistogram(150, 20)
            r.test()
        if args[1] == "ff":
            f = FindFeatures(150, 20, 0, 2)
            #f.plot_mean_var()
            f.plot_mean_var_diff()
            #f.plot_mean()
            #f.plot_var()
        if args[1] == "sf":
            f = FindFeatures(150, 20, 0, 2)
            f.plot_significant()
        if args[1] == "test":
            f = FindFeatures(150, 20, 10)
            train = f.get_significant('train')
            test = f.get_significant('test')
            print(np.shape(train))
            print(np.shape(test))
        if args[1] == "predict":
            predict(150, 20, 0, 2)

    print("Program execution took %s seconds" %round(time.time() - start_time, 3))
