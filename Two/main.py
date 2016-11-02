"""Main file."""

import sys
import os
from time import strftime

from app.trial_one import submission_predictor

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