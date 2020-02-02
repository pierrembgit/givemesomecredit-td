import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import mlflow
# import mlflow.sklearn

def main():
    GMSC_train_data = pd.read_csv('sources/cs-training.csv', index_col=0)
    # GMSC_train_data.describe()
    Y_train = GMSC_train_data['SeriousDlqin2yrs']
    GMSC_train_data.drop(columns=['SeriousDlqin2yrs'], inplace=True)
    X_train = GMSC_train_data

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    X_train_std = num_pipeline.fit_transform(X_train)

    skf = StratifiedKFold(
        n_splits = 10,
        shuffle = True,
        random_state = 42
    )

    xgb = XGBClassifier(
        booster = 'gbtree',
        objective = 'binary:logistic',
        nthread = 8
    )
    
    params = {
        'min_child_weight': [5, 6, 7, 8, 9, 10],
        'gamma': [0.4, 0.5, 0.6, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
        'max_depth': [4, 5, 6],
        'learning_rate' : [0.05]
    }

    search_cv = RandomizedSearchCV(
        estimator = xgb,
        param_distributions = params,
        scoring = 'roc_auc',
        n_iter = 2,
        n_jobs = 8,
        # cv = skf.split(X_train_std, Y_train),
        cv = 5,
        verbose = 3,
    )

    search_cv.fit(X_train_std, Y_train)

    # log parameters, metrics, and model
    mlflow.start_run()
    mlflow.log_params(params)
    mlflow.log_metrics({'roc_auc': search_cv.best_score_})
    # mlflow.xgboost.log_model(xgb, 'model', args.conda_env)
    mlflow.xgboost.log_model(xgb, 'model')
    print('Model logged in run {}'.format(run.info.run_uuid))


if __name__ == '__main__':
    main()