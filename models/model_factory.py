from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(model_name, **kwargs):

    if model_name == "logistic":
        return LogisticRegression(
            max_iter = 1000,
            solver = "lbfgs"
        )
    
    elif model_name == "rf":
        max_depth = kwargs.get("max_depth")
        if max_depth is None:
            max_depth = 6
        n_jobs = kwargs.get("n_jobs", -1)

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=max_depth,
            random_state=42,
            n_jobs=n_jobs
        )
    
    elif model_name == "xgb":
        max_depth = kwargs.get("max_depth")
        if max_depth is None:
            max_depth = 4
            
        return XGBClassifier(
            n_estimators=300,
            max_depth=max_depth,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")