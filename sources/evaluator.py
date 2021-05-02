# System modules
import pandas as pd
from sklearn.metrics import fbeta_score
from skopt import BayesSearchCV
from skopt.callbacks import DeltaXStopper

# Global variables
BETA = 2


class Evaluator:

    def __init__(self, x_test, y_test):
        self.estimator = None
        self.x_test = x_test
        self.y_test = y_test

    def scoring_function(self, estimator, args=None):
        '''
        This scoring function favors recall because false
        negatives are more harmful than false positives.
        '''
        y_pred = estimator.predict(self.x_test)
        score = fbeta_score(self.y_test, y_pred, beta=BETA)
        return score

    def train(self, estimator, params, x_train):

        search = BayesSearchCV(
            estimator,
            params,
            scoring=self.scoring_function,
            cv=5,
            n_iter=20,
            return_train_score=True,
            random_state=1
        )
        results = search.fit(x_train, callback=DeltaXStopper(1e-8))
        self.estimator = results.best_estimator_

        return results

    def predict(self, x):
        y_pred = self.estimator.predict(x)
        return y_pred
