# System modules
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from skopt.space import Integer, Real


CONFIG = {
    'k-Nearest Neighbors': {
        'estimator': KNN(),
        'params': {
            'n_neighbors': Integer(5, 50),
        }
    },
    'Local Outlier Factor': {
        'estimator': LOF(),
        'params': {
            'n_neighbors': Integer(5, 50),
            'leaf_size': Integer(10, 50)
        }
    },
    'One-Class SVM': {
        'estimator': OCSVM(),
        'params': {
            'gamma': Real(1e-6, 1e+6, prior='log-uniform'),
            'nu': Real(1e-6, (1 - 1e-6))
        }
    }
}
