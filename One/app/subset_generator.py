from itertools import chain, combinations
import numpy as np

def all_subsets(ss, minimum):
    return sum([list(map(list, combinations(ss, i))) for i in range(minimum,len(ss)+1)], [])

