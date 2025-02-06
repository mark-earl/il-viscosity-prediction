# XGBoost `objective` Performance Analysis
*NOTE: All objectives run with default hyperparameters*
| Metric | CI | Average R^2 |
|-|-|-|
|`reg:gamma` - *Gamma Regression With Log-Link*|(.81, .82)|.81
|`reg:squarederror` - *Regression With Squared Loss*|(.80, .82)|.81
|`reg:tweedie` - *Tweedie Regression With Log-Link*|(.80, .82)|.81
|`reg:absoluteerror` - *Regression with L1 Error*|(.77, .79)|.78
|`reg:quantileerror` - *Quantile Loss w/ alpha=0.5*|(.77, .79)|.78
|`reg:squaredlogerror` - *Regression With Squared Log Loss*|(.74, .75)|.74
|`reg:pseudohubererror` - *Regression With Pseudo Huber Loss*|(-3572.11, -3393.12)|-3482.61
