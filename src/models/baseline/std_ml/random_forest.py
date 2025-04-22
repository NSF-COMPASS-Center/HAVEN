import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import random
from utils import utils


def run(X_train, X_test, y_train, rf_settings):
    rf_model = RandomForestClassifier(class_weight="balanced")
    classification_type = rf_settings["classification_type"]
    if classification_type == "multi":
        print("Multiclass Random Forest Model")
        # multinomial: multi-class cross-entropy loss

    # K-Fold Cross Validation: START #
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5;
    # shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"n_estimators": rf_settings["n_estimators"], "max_depth": rf_settings["max_depth"]} #, "max_features": rf_settings["max_features"]}
    scoring_param = "accuracy"
    print(f"Tuning hyper-parameters {tuning_parameters} based on {scoring_param}")

    # use stratified k-fold to ensure each set contains approximately the same percentage of samples of each target class as the complete set.
    # TODO: should we use StratifiedShuffleSplit instead? What is the difference?
    kfold_cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    # refit=True : retrain the best model_params on the full training dataset
    cv_model = GridSearchCV(estimator=rf_model, param_grid=tuning_parameters, scoring=scoring_param,
                            cv=kfold_cv_model, verbose=2, return_train_score=False, refit=True)
    cv_model.fit(X_train, y_train)

    # The best values chosen by KFold-cross-validation
    print("Best parameters in trained model_params = ", cv_model.best_params_)
    print("Best score in trained model_params = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    # K-Fold Cross Validation: END #

    y_pred = classifier.predict_proba(X_test)
    # classifier.feature_importances_ : ndarray of shape (n_features, 1)
    # convert to pandas dataframe of shape (1, n_features)
    feature_names = classifier.feature_names_in_
    n_features = len(feature_names)
    feature_importances = pd.DataFrame(classifier.feature_importances_.reshape(1, n_features), columns=classifier.feature_names_in_)
    validation_scores = utils.get_validation_scores(cv_model.cv_results_)

    return y_pred, feature_importances, validation_scores, classifier


