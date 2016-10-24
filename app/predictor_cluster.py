"""Description of this file."""


import os
import sys

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from scipy.stats import linregress

import itertools

from .load_data import load_targets, load_samples_inputs

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso, Huber
from sklearn.cross_validation import cross_val_predict
from sklearn.decomposition import IncrementalPCA
from .histogram_plot import histogram_plot

from .settings import CURRENT_DIRECTORY


__author__ = "lfievet"
__copyright__ = "Copyright 2016, Project One"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "19/10/2016"
__status__ = "Production"


def predict_cluster():
    get_pixel_age_correlating()
    # predict_training()
    return

    train, tag = get_brain_pca_train(0, 100)
    test, tag = get_brain_pca_test(0, 100)
    test_young, tag = get_brain_pca_test(0, 60)
    test_old, tag = get_brain_pca_test(61, 100)

    cols = [0, 1, 2, 3]
    xs = train[cols].values.tolist()
    ys = train["Y"].tolist()

    young_old = [0 if y <= 60 else 1 for y in ys]

    predictor = KNeighborsClassifier()
    predictor.fit(xs, young_old)

    xs_test = test[cols].values.tolist()
    age_group = predictor.predict(xs_test)
    test["G"] = age_group
    test_young["G"] = age_group
    test_old["G"] = age_group
    print(age_group)

    # Split test set into young and old
    young_test = test_young[test_young["G"] == 0]
    old_test = test_old[test_old["G"] == 1]

    # Predict by age group
    young, tag = get_brain_pca_train(0, 60)
    old, tag = get_brain_pca_train(61, 100)

    young_test["Prediction"] = predict(young, young_test)
    old_test["Prediction"] = predict(old, old_test)

    # print(young_test.head())
    # print(old_test.head())

    test_full = young_test.append(old_test)
    test_full.sort_index(inplace=True)
    print(test_full)

    test_full["ID"] = test_full.index.values + 1
    test_full = test_full[["ID", "Prediction"]]

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "data",
        "predictions.csv"
    )
    test_full.to_csv(prediction_path, index=False)


def predict(train, test):
    cols = [0, 1, 2]
    train_xs = train[cols].values.tolist()
    train_ys = train["Y"].tolist()
    predictor = GradientBoostingRegressor()
    predictor.fit(train_xs, train_ys)

    test_xs = test[cols].values.tolist()
    predicted = predictor.predict(test_xs)
    return predicted


def predict_training():
    data, tag = get_brain_pca_train(0, 100)

    ages = data["Y"].tolist()
    histogram_plot(ages, "ages")

    cols = [0, 1, 2, 3]
    # refine = 17
    xs = data[cols].values.tolist()
    ys = data["Y"].tolist()

    young_old = [0 if y <= 60 else 1 for y in ys]

    predictor = KNeighborsClassifier()
    predicted = cross_val_predict(predictor, xs, young_old, cv=5)
    print("")
    print(predicted)
    mse(young_old, predicted)

    young, tag = get_brain_pca_train(0, 60)
    old, tag = get_brain_pca_train(61, 100)
    cross_predict(young)
    cross_predict(old)

    print(young)
    print(old)

    ages = []
    diffs = []
    predicted_ages = []
    for idx, p in enumerate(predicted):
        actual_age = data.ix[idx]["Y"]

        if p == 0 and idx in young.index.values:
            predicted_age = young.ix[idx]["P"]
        elif p == 1 and idx in old.index.values:
            predicted_age = old.ix[idx]["P"]
        else:
            continue

        ages.append(actual_age)
        predicted_ages.append(predicted_age)
        diffs.append(predicted_age - actual_age)

    # return

    # predictor = KNeighborsRegressor(
    #     n_neighbors=4,
    #     weights="uniform",
    #     p=2,
    # )
    # predictor = Lasso()
    # predicted = cross_val_predict(predictor, xs, ys, cv=5)

    # for idx, p in enumerate(predicted):
    #     data_r = data[data.index != idx]
    #     data_r = data_r[data_r["Y"] <= p + refine]
    #     data_r = data_r[p - refine <= data_r["Y"]]
    #     xs_r = data_r[cols].values.tolist()
    #     ys_r = data_r["Y"].values.tolist()
    #     predictor = Lasso()
    #     predictor.fit(xs_r, ys_r)
    #     inputs = data[cols].ix[idx].tolist()
    #     predicted[idx] = predictor.predict(inputs)
    #     print("{}/{}".format(p, predicted[idx]))

    # diffs = predicted - ys
    # mse(ages, predicted)
    print(len(diffs))
    print(np.mean(list(map(lambda x: x*x, diffs))))

    plt.figure()
    plt.scatter(
        ages,
        predicted_ages,
    )
    plt.plot(
        np.linspace(0, 100),
        np.linspace(0, 100),
    )
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.savefig("plots/diffs-{}.pdf".format(tag))


def cross_predict(data):
    cols = [0, 1, 2]
    xs = data[cols].values.tolist()
    ys = data["Y"].tolist()
    predictor = GradientBoostingRegressor()
    predicted = cross_val_predict(predictor, xs, ys, cv=5)
    data["P"] = predicted


def mse(ys, predicted):
    diffs = predicted - ys
    mean_squared_error = np.mean(list(map(lambda x: x*x, diffs)))
    print(mean_squared_error)


def get_brain_pca_test(age_min, age_max):
    tag = "{}-{}".format(
        age_min,
        age_max
    )

    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "pca-{}-test.hdf".format(tag)
    )
    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        # Load the training data
        training_inputs = load_samples_inputs()
        data = load_targets()

        # Select the age range
        data = data[data["Y"] >= age_min]
        data = data[data["Y"] <= age_max]
        indices = data.index.values
        training_inputs = [training_inputs[i] for i in indices]

        # Compute the PCA
        training_mean, pca = get_diff_brain_pca(training_inputs)

        # Load the test inputs
        test_inputs = load_samples_inputs(False)

        # Compute the test brains difference to the mean brain
        test_inputs = get_diff_brain(test_inputs, training_mean)

        # Compute the test transforms
        test_transforms = transform_pca(pca, test_inputs)

        data = pd.DataFrame()
        for i in range(0, len(test_transforms)):
            data[i] = test_transforms[i]

        data = data.transpose()

    data.to_hdf(cache_path, "table")

    return data, tag


def get_brain_pca_train(age_min, age_max):
    tag = "{}-{}".format(
        age_min,
        age_max
    )

    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "pca-{}.hdf".format(tag)
    )
    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        inputs = load_samples_inputs()
        data = load_targets()

        data = data[data["Y"] >= age_min]
        data = data[data["Y"] <= age_max]
        indices = data.index.values
        inputs = [inputs[i] for i in indices]

        data = get_brain_pca(inputs, data, tag)
        data.index = indices

    data.to_hdf(cache_path, "table")

    return data, tag


def get_brain_pca(inputs, data, tag=""):
    l = len(inputs)

    mean = get_mean_brain(inputs)
    pca_inputs = get_diff_brain(inputs, mean)
    ages = data["Y"].tolist()

    pca = fit_pca(pca_inputs)
    pca_values = transform_pca(pca, pca_inputs)

    df = pd.DataFrame()
    for i in range(0, l):
        df[i] = pca_values[i]

    df = df.transpose()
    df["Y"] = ages

    for i in range(0, len(pca.explained_variance_)):
        features = list(map(lambda x: x[i], pca_values))
        correlation_plot(features, ages, "correlation-{}-{}".format(tag, i))

    return df


def get_diff_brain_pca(inputs):
    mean = get_mean_brain(inputs)
    pca_inputs = get_diff_brain(inputs, mean)
    pca = fit_pca(pca_inputs)
    return mean, pca


def transform_pca(pca, inputs):
    l = len(inputs)
    pl = 10

    transforms = []
    for i in range(0, l//pl + 1):
        start = pl * i
        end = min(l, pl * (i+1))

        if end > start:
            values = pca.transform(inputs[start:end])
            for v in values:
                transforms.append(v)

    return transforms


def fit_pca(inputs):
    l = len(inputs)
    pl = 10

    pca = IncrementalPCA(n_components=6, batch_size=10)

    for i in range(0, l//pl + 1):
        start = pl * i
        end = min(l, pl * (i+1))
        if end > start:
            pca.partial_fit(inputs[start:end])

    return pca


def get_diff_brain(inputs, mean):
    for idx, i in enumerate(inputs):
        x = inputs[idx].get_data()[:, :, :, 0] - mean[:, :, :, 0]
        x = x.flatten()
        inputs[idx] = x

    return inputs


def get_mean_brain(inputs):
    l = len(inputs)
    mean = divide(inputs[0], l)

    for i in range(0, l):
        v = divide(inputs[i], l)
        mean = np.add(mean, v)

    return mean


def correlation_plot(features, ages, filename):
    plt.figure()
    plt.scatter(ages, features)
    plt.xlabel('Age')
    plt.ylabel('PCA-1')
    plt.title(r'$\mathrm{PCA-1 as function of age}$')
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))


def divide(sample_input, l):
    return np.divide(sample_input.get_data(), l)


def get_pixel_age_correlating():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "pixel-age-correlating-training.hdf"
    )

    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = pixel_age_correlation_compute()
        data.to_hdf(cache_path, "table")

    print("")
    print(data)

    return data


def pixel_age_correlation_compute():
    df = pd.DataFrame()
    training_inputs = load_samples_inputs()
    data = load_targets()
    ages = data["Y"].tolist()

    inputs = [i.get_data() for i in training_inputs]

    l_x = len(inputs[0][:, 0, 0, 0])
    l_y = len(inputs[0][0, :, 0, 0])
    l_z = len(inputs[0][0, 0, :, 0])

    x_range = range(0, l_x)
    y_range = range(0, l_y)
    z_range = range(0, l_z)

    slopes = []
    rs = []
    ps = []
    count = 0
    for c in itertools.product(x_range, y_range, z_range):
        print_progress("({}, {}, {})".format(*c))

        vs = [i[c[0], c[1], c[2], 0] for i in inputs]

        slope, intercept, r, p, std = linregress(ages, vs)
        slopes.append(slope)
        rs.append(r**2)
        ps.append(p)

        if r**2 > 0.1:
            df[count] = list(c) + vs
            count += 1

    histogram_plot(slopes, "slopes")
    histogram_plot(rs, "rs")
    histogram_plot(ps, "ps")

    df = df.transpose()

    return df


def print_progress(s):
    sys.stdout.write("\r%s" % s)
    sys.stdout.flush()
