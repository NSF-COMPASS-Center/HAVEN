import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
import random
from utils import utils


def run(X_train, X_test, y_train, svm_settings):
    # for multiclass classification svc uses one-vs-one to train n*(n-1)/2 classifiers for each pair of classes
    # we set decision_function_shape="ovr" in the classifier instantiation convert the output decision function from ovo to one-vs-rest format
    # i.e., (n_samples, n_classes) for consistency with all other classifiers

    svm_model = SVC(kernel=svm_settings["kernel"], # supported types: ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
                    class_weight="balanced", # uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data
                    max_iter=-1, # no_limit
                    verbose=True,
                    decision_function_shape="ovr",
                    break_ties=False, # the first class among the tied classes is returned. If True, ties are broken using the confidence values of the decision function
                    random_state=random.randint(0, 10000))

    # K-Fold Cross Validation: START #
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5;
    # shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"C": svm_settings["C"]}
    scoring_param = "accuracy"
    print(f"Tuning hyper-parameters {tuning_parameters} based on {scoring_param}")

    # use stratified k-fold to ensure each set contains approximately the same percentage of samples of each target class as the complete set.
    # TODO: should we use StratifiedShuffleSplit instead? What is the difference?
    kfold_cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    # refit=True : retrain the best model on the full training dataset
    cv_model = GridSearchCV(estimator=svm_model, param_grid=tuning_parameters, scoring=scoring_param,
                            cv=kfold_cv_model, verbose=2, return_train_score=False, refit=True)
    cv_model.fit(X_train, y_train)

    # The best values chosen by KFold-cross-validation
    print("Best parameters in trained model = ", cv_model.best_params_)
    print("Best score in trained model = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    # K-Fold Cross Validation: END #

    y_pred = classifier.predict_proba(X_test)
    validation_scores = utils.get_validation_scores(cv_model.cv_results_)
    feature_importances=None
    return y_pred, feature_importances, validation_scores, classifier


