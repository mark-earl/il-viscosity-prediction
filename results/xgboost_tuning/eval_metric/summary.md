# XGBoost `eval_metric` Performance Analysis
| Metric | CI | Average R^2 |
|-|-|-|
|`rmse` - *Root Mean Square Error*|(.80, .82)|.81
|`rmsle` - *Root Mean Square Log Error*|(.80, .82)|.81
|`mae` - *Mean Absolute Error*|(.80, .82)|.81
|`mape` - *Mean Absolute Percentage Error*|(.80, .82)|.81
|`mphe` - *Mean Pseudo Huber Error*|(.80, .82)|.81
|`logloss` - *Negative Log-Likelihood*|(.80, .82)|.81
|`error` - *Binary Classification Error Rate*|(.80, .82)|.81
|`error@3` - *Binary Classification Error Rate w/ t*|(.80, .82)|.81
|`merror` - *Multiclass Classification Error Rate*|(.80, .82)|.81
|`mlogloss` - *Multiclass Logloss*|(.80, .82)|.81
|`auc` - *Receiver Operating Characteristic Area Under The Curve*|(.80, .82)|.81
|`aucpr` - *Area Under The PR Curve*|(.80, .82)|.81
|`pre` - *Precision at k*|(.80, .82)|.81
|`ndcg` - *Normalized Discounted Cumulative Gain*|(.80, .82)|.81
|`map` - *Mean Average Precision*|(.80, .82)|.81

"The `eval_metric` parameter in XGBoost is used to define the evaluation metric for model training, but it does not directly affect the final model's performance or its R² score."
