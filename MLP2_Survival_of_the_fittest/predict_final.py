"""Main file."""

import time

from src.submission1 import submission_predictor

if __name__ == "__main__":
    start_time = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S"))

    submission_predictor()

    print("Program execution took %s seconds" %round(time.time() - start_time, 3))
