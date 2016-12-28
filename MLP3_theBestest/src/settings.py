"""Settings file."""

import os

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

DATA_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "data")

CACHE_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "cache")

PLOT_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "plots")

ITERATE_DIRECTORY = os.path.join(CURRENT_DIRECTORY, "..", "iterate")
