from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import random


def run(X_train, X_test, y_train, lr_settings):
    print(y_train)
    lr_model = LogisticRegression(solver="saga", max_iter=1000, penalty="l1")

    ## K-Fold Cross Validation: START ##
    # hyper-parameter tuning using K-Fold Cross Validation with K = 5; shuffle the data with given random seed before splitting into batches
    tuning_parameters = {"C": lr_settings['c']}
    evaluation_params = ["average_precision"]
    kfold_cv_model = KFold(n_splits=5, shuffle=True, random_state=random.randint(0, 10000))

    for evaluation_param in evaluation_params:
        print("\nTuning hyper-parameters based on %s" % evaluation_param)
        cv_model = GridSearchCV(estimator=lr_model, param_grid=tuning_parameters, scoring=evaluation_param,
                                cv=kfold_cv_model, verbose=2, return_train_score=True)
        cv_model.fit(X_train, y_train)

        # The best values chosen by KFold-cross-validation
        print("Best parameters in trained model = ", cv_model.best_params_)
        print("Best score in trained model = ", cv_model.best_score_)
    classifier = cv_model.best_estimator_
    ## K-Fold Cross Validation: END ##

    ## TRAINING: START ##
    print("Training best model from k-fold cross validation over full training set")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict_proba(X_test)
    return y_pred
    ## TRAINING: END ##