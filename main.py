"""Main file."""

import sys
import os

from app.load_data import load_sample_input
from app.heatmap import heatmap
from app.predictor import predict
from app.Checking import checkdata

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

# If the python interpreter is running that module (the .py file) as the main program, it sets the special __name__
# variable to have a value "__main__". If this file is being imported from another module, __name__ will be set to the module's name.

if __name__ == "__main__":
    if not os.path.exists(CACHE_DIRECTORY):
        os.makedirs(CACHE_DIRECTORY)

    if not os.path.exists(PLOT_DIRECTORY):
        os.makedirs(PLOT_DIRECTORY)

# syy.argv is a list of command line arguments passed to a Python script.
# argv[0] is the script name (it is operating system dependent whether this is a full pathname or not).

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
        elif args[1] == "check":
            checkdata()

# Depending on the first command line argument (args[1]) the specified function is executed (i.e. load, heatmap, or predict).