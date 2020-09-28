#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# System libraries
import numpy as np
import os
import tensorflow
import warnings
from hyperopt import hp
from pyod.models.auto_encoder import AutoEncoder as BaseAE
from pyod.models.iforest import IsolationForest as BaseIF
from pyod.models.knn import KNN as BaseKNN
from pyod.models.lof import LocalOutlierFactor as BaseLOF
from pyod.models.ocsvm import OCSVM as BaseOCSVM
from pyod.models.pca import PCA as BasePCA
from scipy.stats import iqr

# Configurations
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
np.random.seed(1)
tensorflow.random.set_seed(1)


def _get_threshold(scores):
    return np.median(scores) + 1.5 * iqr(scores)


class AutoEncoder(BaseAE):
    PARAMS = {"hidden_neurons": hp.randint("hidden_neurons", 5, 20),
              "hidden_layers": hp.randint("hidden_layers", 1, 10),
              "epochs": hp.randint("epochs", 50, 200),
              "batch_size": hp.randint("batch_size", 10, 100),
              "dropout_rate": hp.uniform("dropout_rate", 0, 1)}

    def __init__(self, hidden_neurons=10, hidden_layers=2, epochs=100,
                 batch_size=32, dropout_rate=0.2):

        # Create layers
        first_layers = []
        for i in range(hidden_layers):
            neurons = int(hidden_neurons / (i+1))
            first_layers.append(neurons)
        last_layers = list(reversed(first_layers))
        layers = first_layers + last_layers

        super().__init__(hidden_neurons=layers, epochs=epochs, batch_size=batch_size,
                         dropout_rate=dropout_rate, random_state=1, verbose=0)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        BaseAE.fit(self, x)
        self.model_threshold_ = self.threshold_.copy()

    def predict(self, x):
        scores = BaseAE.decision_function(self, x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions


class IsolationForest(BaseIF):
    PARAMS = {"n_estimators": hp.randint("n_estimators", 50, 400)}

    def __init__(self, n_estimators=100):
        self.model_ = BaseIF(
            n_estimators=n_estimators, random_state=1, contamination=0.1)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        self.model_.fit(x)
        scores = self.decision_function(x)
        self.model_threshold_ = _get_threshold(scores)

    def predict(self, x):
        scores = self.model_.decision_function(x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions

    def decision_function(self, x):
        scores = self.model_.score_samples(x) * -1
        return scores

    def get_norm_scores(self, x):
        scores = self.decision_function(x)
        scores_norm = scores / self.model_threshold_
        return scores_norm


class KNN(BaseKNN):
    PARAMS = {"n_neighbors": hp.randint("n_neighbors", 3, 20),
              "radius": hp.uniform("radius", 0.5, 2)}

    def __init__(self, n_neighbors=5, radius=1):
        BaseKNN.__init__(self, n_neighbors=n_neighbors, radius=radius)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        BaseKNN.fit(self, x)
        scores = BaseKNN.decision_function(self, x)
        self.model_threshold_ = _get_threshold(scores)

    def predict(self, x):
        scores = BaseKNN.decision_function(self, x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions

    def get_norm_scores(self, x):
        scores = BaseKNN.decision_function(self, x)
        scores_norm = scores / self.model_threshold_
        return scores_norm


class LocalOutlierFactor(BaseLOF):
    PARAMS = {"n_neighbors": hp.randint("n_neighbors", 3, 20),
              "leaf_size": hp.randint("leaf_size", 10, 50)}

    def __init__(self, n_neighbors=20, leaf_size=30):
        self.model_ = BaseLOF(n_neighbors=n_neighbors, leaf_size=leaf_size,
                              novelty=True, contamination=0.1)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        self.model_.fit(x)
        scores = self.decision_function(x)
        self.model_threshold_ = _get_threshold(scores)

    def predict(self, x):
        scores = self.decision_function(x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions

    def decision_function(self, x):
        scores = self.model_.score_samples(x) * -1
        return scores

    def get_norm_scores(self, x):
        scores = self.decision_function(x)
        scores_norm = scores / self.model_threshold_
        return scores_norm


class OneClassSVM(BaseOCSVM):
    PARAMS = {"gamma": hp.uniform("gamma", 0, 5),
              "nu": hp.uniform("nu", 0, 0.99)}

    def __init__(self, gamma="auto", nu=0.5):
        super().__init__(gamma=gamma, nu=nu)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        BaseOCSVM.fit(self, x)
        scores = BaseOCSVM.decision_function(self, x)
        self.model_threshold_ = _get_threshold(scores)

    def predict(self, x):
        scores = BaseOCSVM.decision_function(self, x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions

    def get_norm_scores(self, x):
        scores = BaseOCSVM.decision_function(self, x)
        scores_norm = scores / self.model_threshold_
        return scores_norm


class PCA(BasePCA):
    PARAMS = {"n_components": hp.uniform("n_components", 0.5, 0.99)}

    def __init__(self, n_components=None):
        super().__init__(n_components=n_components, random_state=1)
        self.model_threshold_ = None

    def fit(self, x, y=None):
        BasePCA.fit(self, x)
        scores = BasePCA.decision_function(self, x)
        self.model_threshold_ = _get_threshold(scores)

    def predict(self, x):
        scores = BasePCA.decision_function(self, x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions

    def get_norm_scores(self, x):
        scores = BaseOCSVM.decision_function(self, x)
        scores_norm = scores / self.model_threshold_
        return scores_norm


class Ensemble:
    MODELS = {
        "KNN": KNN,
        "LocalOutlierFactor": LocalOutlierFactor,
        "OneClassSVM": OneClassSVM,
    }

    def __init__(self):
        self.models_ = []
        self.model_threshold_ = []

    def fit(self, x):
        scores = np.zeros(len(x))
        for model in self.models_:
            scores += model.get_norm_scores(x)
        self.model_threshold_ = _get_threshold(scores)

    def stacking(self, x):
        scores = np.zeros((len(x), len(self.models_)))
        for i, model in enumerate(self.models_):
            scores[:, i] = model.get_norm_scores(x)

    def predict(self, x):
        scores = np.zeros(len(x))
        for model in self.models_:
            scores += model.get_norm_scores(x)
        predictions = np.where(scores <= self.model_threshold_, 0, 1)
        return predictions
