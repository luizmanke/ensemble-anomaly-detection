#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System libraries
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hyperopt import fmin, tpe
from joblib import delayed, Parallel
from sklearn.metrics import auc, accuracy_score, f1_score, jaccard_score, roc_curve
from sklearn.preprocessing import RobustScaler

# Own libraries
import dataset_reader as dr
import models
from samples_generator import SelfAdaptiveShifting

# Configurations
np.random.seed(1)

# Globals
DATASETS = {"IMS": dr.ims,
            "Turbofan": dr.turbofan}
MODELS = {
    # "AutoEncoder": models.AutoEncoder,
    # "IsolationForest": models.IsolationForest,
    "KNN": models.KNN,
    "LocalOutlierFactor": models.LocalOutlierFactor,
    "OneClassSVM": models.OneClassSVM,
    # "PCA": models.PCA,
    "Ensemble": models.Ensemble,
}


# TODO: Refactor sample creation
def run():
    results = {}

    # Compute cases
    cases, labels = _get_data(dataset_name="Turbofan")
    results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(_compute_case)(i, data) for i, data in enumerate(cases))

    # Concatenate results
    results = {}
    for model_name in results_list[0]:
        results[model_name] = {"elapsed_time": 0, "label_pred": [], "life_period": []}
    for result in results_list:
        for model_name, items in result.items():
            results[model_name]["elapsed_time"] += items["elapsed_time"]
            results[model_name]["label_pred"].append(items["label_pred"])
            results[model_name]["life_period"].append(items["life_period"])

    # Compute indicators
    for model_name in results:
        model_results = results[model_name]
        model_indicators = _compute_indicators(model_results, labels)
        _save_results(model_name, model_indicators)


def _get_data(dataset_name):
    cases, labels = DATASETS[dataset_name]()
    return cases, labels


def _compute_case(case, data):
    data = data.values

    # Split data
    x_train = _split_data(data)

    # Normalization
    scaler = RobustScaler()
    x_train_norm = scaler.fit_transform(x_train)
    data_norm = scaler.transform(data)

    # Create samples
    x_pseudo, y_pseudo = _create_samples(x_train_norm)

    # Loop models
    results = {}
    scores = {}
    for model_name in MODELS:

        # Create model
        model, elapsed_time = _create_model(model_name, x_train_norm, x_pseudo, y_pseudo)

        # Predict
        predictions, scores[model_name] = _predict(model, data_norm)
        label_pred, life_period = _find_anomalies(predictions)

        # Append
        results[model_name] = {
            "elapsed_time": elapsed_time,
            "label_pred": label_pred,
            "life_period": life_period
        }

    # Plot
    if case % 35 == 0:
        plt.figure()
        plt.hlines(1, 0, len(data), linestyle="--", colors="red")
        for model_name, points in scores.items():
            plt.plot(points, label=model_name)
        plt.legend(loc="best")
        plt.savefig(f"figures/case_{case}.png")
        plt.close()

    return results


def _split_data(data):
    USEFUL_PERCENTAGE = 0.3
    size = data.shape[0]
    useful_size = int(size * USEFUL_PERCENTAGE)
    useful_data = data[:useful_size]
    return useful_data


def _create_samples(data):
    generator = SelfAdaptiveShifting(data)
    generator.edge_pattern_detection()

    x_outliers = generator.generate_pseudo_outliers()
    y_outliers = np.ones(x_outliers.shape[0])

    x_targets = generator.generate_pseudo_targets()
    y_targets = np.zeros(x_targets.shape[0])

    pseudo_data = np.concatenate([x_outliers, x_targets])
    pseudo_labels = np.concatenate([y_outliers, y_targets])

    return pseudo_data, pseudo_labels


def _create_model(model_name, x_train, x_test, y_test):
    initial_time = dt.datetime.now()

    def get_best_model(model_name):
        best_params = fmin(fn=evaluate, space=model_class.PARAMS,
                           algo=tpe.suggest, max_evals=50,
                           rstate=np.random.RandomState(1),
                           verbose=0)
        best_model = create_model(**best_params)
        return best_model

    def evaluate(kwargs):
        model = create_model(**kwargs)
        predictions = model.predict(x_test)
        fpr, tpr, _ = roc_curve(y_test, predictions)
        roc_auc_score = auc(fpr, tpr)
        return -1 * roc_auc_score

    def create_model(**kwargs):
        model = model_class(**kwargs)
        model.fit(x_train)
        return model

    # Train
    if model_name != "Ensemble":
        model_class = MODELS[model_name]
        best_model = get_best_model(model_name)
    else:
        best_model = MODELS[model_name]()
        for _, submodel_class in MODELS[model_name].MODELS.items():
            model_class = submodel_class
            best_model.models_.append(get_best_model(model_name))
        best_model.fit(x_train)

    elapsed_time = dt.datetime.now() - initial_time
    return best_model, elapsed_time.total_seconds()


def _predict(model, data):
    predictions = model.predict(data)
    scores = model.get_norm_scores(data)
    return predictions, scores


def _find_anomalies(predictions):
    THRESHOLD = 0.5

    # Find anomalies
    predictions = pd.DataFrame(predictions)
    predictions_mean = predictions.rolling(window=20).mean()
    anomalies = predictions_mean[predictions_mean > THRESHOLD]
    anomalies = anomalies.dropna()

    # Save variables
    label_pred = 0
    life_period = float("nan")
    if not anomalies.empty:
        label_pred = 1
        anomaly_index = anomalies.index[0]
        life_period = anomaly_index / predictions.shape[0]

    return label_pred, life_period


def _compute_indicators(results, labels):
    indicators = {"elapsed_time": results["elapsed_time"] / len(labels)}

    # Accuracy
    labels_pred = results["label_pred"]
    indicators.update({
        "accuracy": accuracy_score(labels, labels_pred),
        "f1": f1_score(labels, labels_pred),
        "jaccard": jaccard_score(labels, labels_pred)
    })

    # Remaining useful life
    life_period = results["life_period"]
    remaining_useful_life = []
    for one_remaining_useful_life, one_label in zip(life_period, labels_pred):
        if one_label == 1:
            remaining_useful_life.append(one_remaining_useful_life)
    remaining_useful_life = \
        [float("nan")] if remaining_useful_life == [] else remaining_useful_life
    indicators.update({
        "remaining_useful_life_min": np.min(remaining_useful_life),
        "remaining_useful_life_mean": np.mean(remaining_useful_life),
        "remaining_useful_life_median": np.median(remaining_useful_life),
        "remaining_useful_life_max": np.max(remaining_useful_life),
        "remaining_useful_life_std": np.std(remaining_useful_life)
    })

    return indicators


def _save_results(model_name, indicators):

    # Compute indicators
    indicators = pd.Series(indicators)
    indicators.to_csv(f"results/{model_name}.csv")

    # Compare
    try:
        comparisson = pd.read_csv("results/comparisson.csv", index_col=0)
    except Exception:
        comparisson = pd.DataFrame()
    comparisson[model_name] = indicators
    comparisson.to_csv("results/comparisson.csv")


if __name__ == "__main__":

    # Run
    run()
