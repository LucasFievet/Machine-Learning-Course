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
from sklearn.svm import LinearSVC, SVC
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
    get_age_pixel_correlation()
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


def cross_predict(data, cols=[0, 1, 2], predictor=GradientBoostingRegressor):
    xs = data[cols].values.tolist()
    ys = data["Y"].tolist()
    predictor = predictor()
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


def correlation_plot(features, ages, filename, y_label='PCA-1'):
    plt.figure()
    plt.scatter(ages, features)
    plt.xlabel('Age')
    plt.ylabel(y_label)
    plt.xlim([15, 95])
    # plt.ylim([0, 1])
    plt.title(r'$\mathrm{{{0} as function of age}}$'.format(y_label))
    plt.grid(True)
    plt.savefig("plots/{}.pdf".format(filename))


def divide(sample_input, l):
    return np.divide(sample_input.get_data(), l)


def get_age_pixel_correlation():
    predict_brain_region_dist()
    return

    training = get_pixel_age_correlating()
    data = load_targets()
    ages = data["Y"].tolist()

    test = get_test_age_pixels()

    training = training.iloc[::1, 3:]

    predicted = []
    for test_i in range(3, len(test.columns)):
        print(test_i)

        correlation = training.corrwith(test[test_i])
        correlation = pd.DataFrame(correlation)
        correlation["Age"] = ages

        tested_ages = []
        age_correlations = []
        age_range = 12
        for i in range(min(ages), max(ages) - age_range):
            age_correlation = correlation[correlation["Age"] >= i]
            age_correlation = age_correlation[age_correlation["Age"] <= i+age_range]

            tested_ages.append(i+age_range/2)
            age_correlations.append(age_correlation[0].mean())

        age_index = age_correlations.index(max(age_correlations))
        predicted.append(tested_ages[age_index])
        correlation_plot(age_correlations, tested_ages, "prediction-{}".format(
            test_i
        ))

        print(tested_ages[age_index])

    predictions = pd.DataFrame()
    predictions["Prediction"] = predicted
    predictions["ID"] = range(1, len(predictions)+1)

    prediction_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "data",
        "predictions-2.csv"
    )
    predictions[["ID", "Prediction"]].to_csv(prediction_path, index=False)


def predict_brain_region_dist():
    data = get_brain_regions()
    # data = data[[0, 4, 5, 6, 9, 10, "Y"]]

    for i in range(1, 12):
        vs = data[i]/data[0]
        correlation_plot(vs.tolist(), data["Y"].tolist(), "ratio-{}".format(i))

    print(data)
    # return

    old = [0 if y < 65 else 1 for y in data["Y"].tolist()]
    data["Old"] = old

    xs = data[[0, 4, 5, 6, 9, 10]].values.tolist()
    ys = data["Old"].tolist()

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    predictor = LinearDiscriminantAnalysis()
    predicted = cross_val_predict(predictor, xs, ys, cv=10)
    data["P"] = predicted

    print(np.sum(abs(predicted - old)))

    # print(data)


def get_brain_regions():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "brain-regions-dist-training.hdf"
    )

    if os.path.exists(cache_path):
        return pd.read_hdf(cache_path, "table")

    training = get_pixel_age_correlating()
    data = load_targets()
    ages = data["Y"].tolist()

    print(training.as_matrix())
    from sklearn.cluster import KMeans
    cluster = KMeans(n_clusters=3, n_jobs=4)
    cluster.fit(training.as_matrix())
    print(cluster.labels_)
    print(len(cluster.labels_))
    labels = list(set(cluster.labels_))
    print(labels)

    regions = pd.DataFrame()
    col = 0
    for idx, l in enumerate(labels):
        indices = np.where(cluster.labels_ == l)[0]
        data = training.iloc[indices.tolist(), 3:]
        region = data.mean()
        values = region.tolist()

        regions[col] = values
        col += 1

        name = "region-{}-{}px".format(
            idx,
            len(indices)
        )
        correlation_plot(values, ages, name, name)

        # print(regions)
        means = []
        stds = []
        skews = []
        for i in range(3, 281):
            from scipy.stats import skewnorm
            params = skewnorm.fit(data[i].tolist())
            # # params = histogram_plot(data[280].tolist(), name + "-hist")
            # print(params)
            # means.append(0)
            means.append(params[1])
            stds.append(params[2])
            skews.append(params[0])

        regions[col] = means
        col += 1
        regions[col] = stds
        col += 1
        regions[col] = skews
        col += 1

        correlation_plot(means, ages, name + "-mean")
        correlation_plot(stds, ages, name + "-std")
        correlation_plot(skews, ages, name + "-skew")
        print()
        print("-" * 100)

    # training = training.reset_index()
    # print(training)
    # return
    # regions = training.iloc[:, 4:].transpose()
    # regions = regions.transpose()
    # print(regions)
    # print(len(ages))
    regions["Y"] = ages
    print(regions)

    regions.to_hdf(cache_path, "table")

    cross_predict(
        regions,
        list(range(0, 3)),
        Lasso
    )

    predicted = np.array(regions["P"].tolist())
    actual = np.array(regions["Y"].tolist())

    correlation_plot(
        np.array(predicted),
        np.array(actual),
        "region-predicted-vs-actual"
    )

    mse(actual, predicted)

    return regions

    # slopes = []
    # rs = []
    # ps = []
    # region1 = pd.DataFrame()
    # for i in range(0, len(training)):
    #     print_progress("{}".format(i))
    #
    #     vs = training.iloc[i, 3:].tolist()
    #
    #     slope, intercept, r, p, std = linregress(ages, vs)
    #
    #     if r**2 > 0.55:
    #         slopes.append(slope)
    #         rs.append(r**2)
    #         ps.append(p)
    #         region1 = region1.append(training.ix[i])
    #         print("\n{}: {} - {}".format(
    #             len(region1),
    #             slope,
    #             r**2
    #         ))
    #         # correlation_plot(vs, ages, "age-correlation-{}".format(
    #         #     i
    #         # ))
    #
    # histogram_plot(slopes, "slopes2")
    # histogram_plot(rs, "rs2")
    # histogram_plot(ps, "ps2")
    #
    # cache_path = os.path.join(
    #     CURRENT_DIRECTORY,
    #     "..",
    #     "cache",
    #     "brain-regions-0.55.hdf"
    # )
    # region1.to_hdf(cache_path, "table")

    # correlations = []
    # region1 = pd.DataFrame()
    # first = training.iloc[0, 3:]
    # region1.append(first)
    # c = 100000
    # i = 0
    # while i < len(training):
    #     # print(training.iloc[[0, i], 3:])
    #     # print(training.iloc[[0, i], 3:].corr())
    #     corr = training.iloc[[c, i], 3:].transpose().corr().min().min()
    #     print_progress("{}; {}: {}".format(len(region1), i, corr))
    #     # print(corr)
    #     if corr > 0.8:
    #         region1 = region1.append(training.iloc[i, 3:])
    #
    #     # j = 1
    #     # corr = 1.0
    #     # while corr > 0.9 and i+j < len(training):
    #     #     j += 1
    #     #     rows = training.iloc[i:i+j, 3:].transpose()
    #     #     # print(rows.corr().ix[0, 1])
    #     #     # return
    #     #     # row_next = training.iloc[i+1, 3:].tolist()
    #     #     # corr = np.corrcoef(row, row_next)
    #     #     corr = rows.corr().min().min()
    #     #     print_progress("{}-{}: {}".format(i, j, corr))
    #     #
    #     # correlations.append(corr)
    #     #
    #     # j -= 1
    #     # region1[i] = training.iloc[i:i+j, :].mean()
    #     # i += j
    #     i += 1
    #
    #     # if corr > 0.9:
    #     #     print("merge")
    #     #     df_reduced[i] = training.iloc[i:i+2, :].mean()
    #     #     i += 1
    #     # else:
    #     #     df_reduced[i] = training.iloc[i, :]
    #     #     df_reduced[i+1] = training.iloc[i+1, :]
    #     #
    #     # i += 1
    #
    # # region1 = region1.transpose()
    # print(region1)
    # histogram_plot(correlations, "pairwise_correlations")
    #
    # cache_path = os.path.join(
    #     CURRENT_DIRECTORY,
    #     "..",
    #     "cache",
    #     "brain-regions-training.hdf"
    # )
    # region1.to_hdf(cache_path, "table")

    # training = training.transpose()
    # corr = training.corr()
    # print(corr.min().min())
    # print(corr.max().max())


def get_pixel_age_correlating():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        # "pixel-age-correlating-training.hdf"
        "brain-regions-0.55.hdf"
    )

    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = pixel_age_correlation_compute()
        data.to_hdf(cache_path, "table")

    return data


def get_test_age_pixels():
    cache_path = os.path.join(
        CURRENT_DIRECTORY,
        "..",
        "cache",
        "pixel-age-correlating-test.hdf"
    )

    if os.path.exists(cache_path):
        data = pd.read_hdf(cache_path, "table")
    else:
        data = compute_test_age_pixel()
        data.to_hdf(cache_path, "table")

    return data


def compute_test_age_pixel():
    data = get_pixel_age_correlating()
    xs = data[0].tolist()
    ys = data[1].tolist()
    zs = data[2].tolist()

    test_inputs = load_samples_inputs(False)
    inputs = [i.get_data() for i in test_inputs]

    positions = list(zip(xs, ys, zs))

    df = pd.DataFrame()
    df[0] = xs
    df[1] = ys
    df[2] = zs

    for idx, i in enumerate(inputs):
        print_progress("{}".format(idx))
        vs = [i[p[0], p[1], p[2], 0] for p in positions]
        df[3 + idx] = vs

    return df


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
