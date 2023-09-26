import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from utils import utils
import random


def run(X_train, X_test, y_train, lr_settings):
    lr_model = LogisticRegression(solver="saga", penalty="l1", class_weight="balanced", max_iter=5000)
    classification_type = lr_settings["classification_type"]
    if classification_type == "multi":
        print("Multiclass Logistic Regression Model")
        # multinomial: multi-class cross-entropy loss
        # ovr: one-versus-rest
        lr_model.multi_class = lr_settings["multiclass_type"]
        # for multi_class=ovr, use all cores in the CPU for parallel processing.
        # this settingis ignored and set to default=1 for multi_class=multinomial
        lr_model.n_jobs = -1

    # K-Fold Cross Validation: START #
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5;
    # shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"C": lr_settings["C"]}
    scoring_param = "accuracy"
    print(f"Tuning hyper-parameters {tuning_parameters} based on {scoring_param}")

    # use stratified k-fold to ensure each set contains approximately the same percentage of samples of each target class as the complete set.
    # TODO: should we use StratifiedShuffleSplit instead? What is the difference?
    kfold_cv_model = StratifiedKFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    # refit=True : retrain the best model on the full training dataset
    cv_model = GridSearchCV(estimator=lr_model, param_grid=tuning_parameters, scoring=scoring_param,
                            cv=kfold_cv_model, verbose=2, refit=True)
    cv_model.fit(X_train, y_train)

    # The best values chosen by KFold-cross-validation
    print("Best parameters in trained model = ", cv_model.best_params_)
    print("Best score in trained model = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    # K-Fold Cross Validation: END #

    y_pred = classifier.predict_proba(X_test)
    model_coefficients = get_coefficients(classifier, classification_type)
    validation_scores = utils.get_validation_scores(cv_model.cv_results_)

    return y_pred, model_coefficients, validation_scores, classifier


def get_coefficients(model, classification_type):
    # ndarray of shape (1, n_features) or (n_classes, n_features)
    coefficients = model.coef_
    feature_names = model.feature_names_in_
    if classification_type == "multi":
        classes = model.classes_
        model_coefficients = pd.DataFrame(coefficients, columns=feature_names, index=classes)
    else:
        model_coefficients = pd.DataFrame(coefficients, columns=feature_names)
    return model_coefficients


