"""Main file."""


import sys


from app.load_data import load_data
from app.heatmap import heatmap


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
    args = sys.argv
    if len(args) < 2:
        print("You need to specify a function")
    else:
        if args[1] == "load":
            print(load_data())
        elif args[1] == "heatmap":
            heatmap()
