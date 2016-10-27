import os
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.linear_model import Ridge, Lasso, LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import cross_val_predict
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from .cluster_window import get_count_gray
from .load_data import load_targets, load_samples_inputs
from .settings import CURRENT_DIRECTORY
from .squared_error import squared_error

def simple_ratio():
    y = load_targets()['Y'].tolist()
    train = load_samples_inputs(True)
    test = load_samples_inputs(False)
    ratios_train = np.array(list(map(get_ratios,train)))    
    ratios_test = np.array(list(map(get_ratios,test)))
    print(np.shape(ratios_train))
    print(np.shape(ratios_test))

    comb_names = list(map(list, combinations(['zeros','low','gray','white'], 2)))
    comb_names = list(map(lambda x: x[0]+'-'+x[1], comb_names))
    for i in range(len(ratios_train[0,:])):
        plot(ratios_train.transpose()[i,:],y,comb_names[i],comb_names[i])

    clf = LinearRegression()
    weights = [sqrt(1.0/y.count(y[i])) for i in range(len(y))]
    scaler = preprocessing.StandardScaler().fit(ratios_train)
    Xtrain = scaler.transform(ratios_train)
    Xtest = scaler.transform(ratios_test)
    predicted = cross_val_predict(clf,ratios_train,y,fit_params={'sample_weight':weights},cv=5)
    print(mean_squared_error(y,predicted))

def get_ratios(data):
    data = data.get_data()[:,:,:,0]
    zeros = count_zero(data)
    low = count_range(data,[0,200])
    gray = count_range(data,[650,1000])
    white = count_range(data,[1200,1600])
    combs = list(map(list, combinations([zeros,low,gray,white], 2)))
    ratios = list(map(lambda x: x[1]/x[0], combs))
    return ratios

def count_zero(data):
    return np.count_nonzero(np.logical_not(data>0))

def count_range(data, minmax):
    return np.count_nonzero(np.logical_and(data>minmax[0],data<minmax[1]))
    
def plot(features, ages, filename, y_label, line=False):
    plt.figure()
    plt.scatter(ages, features)
    if line:
        plt.plot(np.linspace(15, 100), np.linspace(15, 100))
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.xlim([15, 95])
    plt.title(r'$\mathrm{{{0} --as function of age}}$'.format(y_label))
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))
