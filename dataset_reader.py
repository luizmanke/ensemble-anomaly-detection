# -*- coding: utf-8 -*-

# System libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configurations
directory = os.path.dirname(__file__)


def fake():
    CENTERS = 10
    N_FEATURES = 20
    N_SAMPLES = 2000
    OUTLIERS_PCT = 0.1
    n_outliers = int(N_SAMPLES * OUTLIERS_PCT)

    # Create data
    inliers, _ = make_blobs(
        n_samples=N_SAMPLES, n_features=N_FEATURES, centers=CENTERS, random_state=1)
    outliers = np.random.uniform(-10, 10, (n_outliers, N_FEATURES))

    # Concatenate data
    data = np.concatenate([inliers, outliers])
    labels = np.concatenate([np.zeros(N_SAMPLES), np.ones(n_outliers)])

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=1)

    return x_train, x_test, y_train, y_test


def ims():
    PATH = f"{directory}/datasets/IMS Bearings"
    SETUPS = {
        "1st_test": {
            "bearing_1": {"channels": [0, 1], "label": 0},
            "bearing_2": {"channels": [2, 3], "label": 0},
            "bearing_3": {"channels": [4, 5], "label": 1},
            "bearing_4": {"channels": [6, 7], "label": 1}},
        "2nd_test": {
            "bearing_1": {"channels": [0], "label": 1},
            "bearing_2": {"channels": [1], "label": 0},
            "bearing_3": {"channels": [2], "label": 0},
            "bearing_4": {"channels": [3], "label": 0}},
        "3rd_test": {
            "bearing_1": {"channels": [0], "label": 0},
            "bearing_2": {"channels": [1], "label": 0},
            "bearing_3": {"channels": [2], "label": 1},
            "bearing_4": {"channels": [3], "label": 0}}}

    # Loop tests
    cases, labels = [], []
    for test_file, bearings in SETUPS.items():
        files = os.listdir(f"{PATH}/{test_file}")

        # Initiate bearings
        new_inputs = {}
        for bearing in bearings:
            new_inputs[bearing] = pd.DataFrame()

        # Load files
        for file in files:
            waveform = pd.read_csv(
                f"{PATH}/{test_file}/{file}", header=None, sep="\t", engine="python")

            # Compute FFT of each channel
            ffts = pd.DataFrame()
            for column in waveform:
                n_samples = waveform[column].shape[0]
                full_fft = scipy.fft.fft(waveform[column].values)
                half_fft = 2.0/n_samples * np.abs(full_fft[0:n_samples//2])
                ffts[column] = half_fft

            # Append to bearings
            for bearing, item in bearings.items():
                new_data = pd.DataFrame()
                for channel in item["channels"]:
                    new_data = pd.concat([new_data, ffts.loc[:, channel]])
                new_data = new_data.transpose()
                dataframe = new_inputs[bearing]
                new_inputs[bearing] = pd.concat([dataframe, new_data])

        # Reduce dimensions
        for bearing, data in new_inputs.items():
            new_data = PCA(n_components=0.8).fit_transform(data)
            new_inputs[bearing] = new_data

        # Append to outputs
        for bearing, new_input in new_inputs.items():
            cases.append(new_input)
            labels.append(bearings[bearing]["label"])

    return cases, labels


# TODO: Use test set
def turbofan():
    PATH = f"{directory}/datasets/Turbofan Engines"
    FILES = ["train_FD001", "train_FD002", "train_FD003", "train_FD004"]

    # Load files
    cases, labels = [], []
    for file in FILES:
        complete_data = pd.read_csv(
            f"{PATH}/{file}.txt", sep="\\s", header=None, engine="python")

        # Split engines
        engines = complete_data.pop(0)
        for engine_number in set(engines):
            subset = engines[engines == engine_number]
            new_case = complete_data.loc[subset.index, :]
            new_case = new_case.sort_values(1).drop(1, axis=1)
            new_case = new_case.reset_index(drop=True)
            new_case.columns = range(new_case.shape[1])

            # Append to outputs
            cases.append(new_case)
            labels.append(1)

    return cases, labels


# TODO: Use test set
def phm08():
    PATH = f"{directory}/datasets/PHM08 Engines"

    # Load files
    complete_data = pd.read_csv(
        f"{PATH}/train.txt", sep="\\s", header=None, engine="python")

    # Split engines
    cases, labels = [], []
    engines = complete_data.pop(0)
    for engine_number in set(engines):
        subset = engines[engines == engine_number]
        new_case = complete_data.loc[subset.index, :]
        new_case = new_case.sort_values(1).drop(1, axis=1)
        new_case = new_case.reset_index(drop=True)
        new_case.columns = range(new_case.shape[1])

        # Append to outputs
        cases.append(new_case)
        labels.append(1)

    return cases, labels


def credit_card():
    PATH = f"{directory}/datasets/Credit Card"
    dataframe = pd.read_csv(f"{PATH}/credit_card.csv")
    classes = dataframe.pop("Class")
    dataframe.pop("Time")
    cases = [dataframe]
    labels = [classes]
    return cases, labels


def connection():
    PATH = f"{directory}/datasets/EECS 498"

    dataframe = pd.read_csv(f"{PATH}/conn250K.csv", header=None)
    dataframe = dataframe.set_index(0)
    dataframe = dataframe.sort_index()
    dataframe = dataframe.reset_index(drop=True)

    classes = pd.read_csv(f"{PATH}/conn250K_anomaly.csv", header=None)
    classes = classes.set_index(0)
    classes = classes.sort_index()
    classes = classes.reset_index(drop=True)
    classes = classes[1]

    cases = [dataframe]
    labels = [classes]
    return cases, labels


def kdd99():
    PATH = f"{directory}/datasets/KDD 99"
    dataframe = pd.read_csv(f"{PATH}/kdd99-unsupervised-ad.csv", header=None)
    classes = dataframe.pop(29)
    classes = classes.replace({"n": 0, "o": 1})
    cases = [dataframe]
    labels = [classes]
    return cases, labels


def letters():
    PATH = f"{directory}/datasets/Letter Recognition"
    dataframe = pd.read_csv(f"{PATH}/letter-unsupervised-ad.csv", header=None)
    classes = dataframe.pop(32)
    classes = classes.replace({"n": 0, "o": 1})
    cases = [dataframe]
    labels = [classes]
    return cases, labels


# TODO: Load Microsoft dataset
def microsoft():
    pass


if __name__ == "__main__":

    # Load
    cases, labels = credit_card()

    # Plot
    dataframe = cases[0]
    classes = labels[0]

    x_norm = StandardScaler().fit_transform(dataframe)
    x_pca = PCA(n_components=2).fit_transform(x_norm)

    plt.figure()
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=classes.values)
    plt.show()
