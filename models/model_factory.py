from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model(model_name):

    if model_name == "logistic":
        return LogisticRegression(
            max_iter = 1000,
            solver = "lbfgs"
        )
    
    elif model_name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_name == "xgb":
        return XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")