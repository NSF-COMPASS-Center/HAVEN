from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import random
from utils import utils

def run(X_train, X_test, y_train, xgb_settings):
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators = 500,
        early_stopping_rounds = 50,
        num_class = xgb_settings['num_classes'],
        objective = 'multi:softmax',
        random_state = random.randint(1, 1000),
        nthread = 1
    )

    # K-Fold Cross Validation: Start #
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5;
    # shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"booster": xgb_settings['booster'],
                         "eta": xgb_settings['eta'],
                         "max_depth": xgb_settings['max_depth'],
                         "subsample": xgb_settings['subsample'],
                         "lambda": xgb_settings['lambda'],
                         "tree_method": xgb_settings['tree_method']}
    scoring_param = "accuracy"
    print(f"Tuning hyper-parameters {tuning_parameters} based on {scoring_param}")

    kfold_cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    cv_model = GridSearchCV(estimator=xgb_model, param_grid=tuning_parameters, scoring=scoring_param,
                            cv=kfold_cv_model, verbose=2, return_train_score=False, refit=True)
    cv_model.fit(X_train, y_train)

    # The best values chosen by KFold-cross-validation
    print("Best parameters in trained model_params = ", cv_model.best_params_)
    print("Best score in trained model_params = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    # K-Fold Cross Validation: END #

    y_pred = classifier.predict_proba(X_test)
    validation_scores = utils.get_validation_scores(cv_model.cv_results_)
    feature_importances = None
    return y_pred, feature_importances, validation_scores, classifier