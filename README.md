# Ensemble Anomaly Detection on Turbofan Engines

Engines play an important part in the industry. Early detecting failures can prevent systems downtime. Unfortunately, not only there are several types of engine failures, but also more than one failure can occur at the same time. Moreover, labeled data is a problem, since it’s often not available.

In this paper, we explore unsupervised anomaly detection approaches on the Turbofan Engines dataset using common machine learning models. To overcome the deficiencies that each model has on its own, a bagging ensemble framework is proposed.

The results for each approach is evaluated using standard accuracy, F1 and Jaccard metrics. In addition, the remaining useful life of each case is also computed. For every metric, the ensemble approach has the highest scores.

The conclusion is that ensemble methods can perform better than individual methods, once the combination of models can make up for each individual drawbacks, since its hyperparameter tuning is also a challenge.

<div style='text-align:center'><img src='images/turbofan.jpg'/></div>

## Development

- [Preprocess](sources/1_preprocess.ipynb)
- [Exploratory Data Analysis](sources/2_eda.ipynb)
- [Modeling](sources/3_modeling.ipynb)

## Results

<blank> | KNN | LocalOutlierFactor | OneClassSVM
--- | --- | --- | ---
f1-score | 0.762 | 0.998 | 0.999

## References
* https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
