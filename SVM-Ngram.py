import numpy as np
import pandas as pd
import scipy as sp
import collections
from collections import Counter
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import csv
import matplotlib.pyplot as plt
import DataOverview as ow
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import calibration
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def cleaned_claims_lst(data, col_name):
    claim_lst = [str(data[col_name][idx]) for idx in data.index]
    cleaned_lst = []
    lemmatizer = WordNetLemmatizer()
    for claim in claim_lst:
        tokens = word_tokenize(claim)
        words = [word.lower() for word in tokens if word.isalpha()]
        cleaned_lst.append(words)
    # print([str(data[col_name][idx]).replace('[^\w\s]', '').lower().split(' ') for idx in data.index])
    return cleaned_lst

def remove_stopWord(textword_list):
    stop_words = set(stopwords.words('english'))
    result = []
    for list in textword_list:
        temp = ""
        for w in list:
            if not w in stop_words:
                temp += " " + w
        result.append(temp)
    return result

def get_corpus(data, col_name):
    claim_lst = cleaned_claims_lst(data, col_name)
    return [word for claim in claim_lst for word in claim]

def use_numbers(claim):
    return int(any(char.isdigit() for char in str(claim)))

def get_length(claim):
    return len(claim.split(" "))

def run_Model(data_train, data_test, y_train, y_true, Optimized=False, color='red'):

    # Baseline
    if not Optimized:
        # data_train, data_test, y_train, y_true = \
        #     train_test_split(data['Claim'], data['NoteTrend'], test_size=0.2)
        ngram_counter = CountVectorizer(analyzer='word')

        X_train = ngram_counter.fit_transform(data_train['Claim'].astype(str))
        X_test = ngram_counter.transform(data_test['Claim'].astype(str))
    # Optimized
    else:
        # data['CleanedClaim'] = remove_stopWord(cleaned_claims_lst(data, 'Claim'))
        # data['UseNumber'] = data.apply(lambda row: use_numbers(row.Claim), axis=1)
        # data['ClaimLength'] = data.apply(lambda row: get_length(row.CleanedClaim), axis=1)


        # data_train, data_test, y_train, y_true = \
        #     train_test_split(data, data['NoteTrend'], test_size=0.2)

        # oversample = RandomOverSampler(sampling_strategy='minority')
        # data_train_over, y_train_over = oversample.fit_resample(data_train, y_train)


        # ngram_counter = CountVectorizer(ngram_range=(1, 2), analyzer='word')
        # X_train = sp.sparse.hstack((ngram_counter.fit_transform(data_train['CleanedClaim'].astype(str)), data_train[['UseNumber']].values),format='csr')
        # X_test = sp.sparse.hstack((ngram_counter.transform(data_test['CleanedClaim'].astype(str)), data_test[['UseNumber']].values),format='csr')
        # # X_columns = ngram_counter.get_feature_names() + data_train[['UseNumber']].columns.tolist()

        mapper = DataFrameMapper([
            (['UseNumber'], None),
            # ('CleanedClaim', CountVectorizer(ngram_range=(1, 2), analyzer='word'))
            ('CleanedClaim', TfidfVectorizer(ngram_range=(1, 2), analyzer='word', norm='l2', min_df=3))
        ])
        X_train = mapper.fit_transform(data_train)
        X_test = mapper.transform(data_test)
        # y_train = y_train_over

    # Fit Model
    # param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                  'C': [1, 10, 100, 1000]},
    #                 {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    # grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=10)
    #
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)


    # clf= SVC(C=10, gamma=0.001, kernel="rbf", probability=True)
    svm = LinearSVC(C=2, dual=True, max_iter=1e6, loss='hinge')
    clf = sklearn.calibration.CalibratedClassifierCV(svm)
    # clf = XGBClassifier(learning_rate=0.1,
    #                       colsample_bytree = 0.5,
    #                       subsample = 0.9,
    #                       objective='binary:logistic',
    #                       n_estimators=50,
    #                       reg_alpha = 0.3,
    #                       max_depth=3,
    #                       gamma=2)

    # Parameter Tuning
    # clf = LogisticRegression()
    # dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    # params = {"objective": "reg:squarederror", "max_depth": 4}
    # cv_results = xgb.cv(dtrain=dmatrix, nfold=3, params=params, metrics="rmse",
    #                     early_stopping_rounds=10, num_boost_round=50, as_pandas=True, seed=123)
    # print(cv_results)
    #
    # # Create the parameter dictionary for each tree (boosting round)
    # params = {"objective": "reg:squarederror", "max_depth": 3}
    # # # Tune eta - learning rate
    # eta_vals = [0.01, 0.1]
    # best_rmse = []
    # for curr_val in eta_vals:
    #     params['eta'] = curr_val
    #
    #     # Perform cross-validation: cv_results
    #     cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=3,
    #                         early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123,
    #                         as_pandas=True)
    #
    #     # Append the final round rmse to best_rmse
    #     best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])
    # print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=['eta', 'best_rmse']))
    #
    # # Create the parameter dictionary
    # params = {"objective": "reg:squarederror"}
    # # Tune max_depth values
    # max_depths = [1,2,3,4,5]
    # best_rmse = []
    # for curr_val in max_depths:
    #     params['max_depth'] = curr_val
    #     cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=2,
    #                         early_stopping_rounds=5, num_boost_round=10, metrics='rmse', seed=123,
    #                         as_pandas=True)
    #     best_rmse.append(cv_results['test-rmse-mean'].tail().values[-1])
    # print(pd.DataFrame(list(zip(max_depths, best_rmse)), columns=['max_depth', 'best_rmse']))
    #
    # # Create the parameter dictionary
    # params = {"objective": "reg:squarederror", "max_depth": 3}
    # # Tune colsample_bytree_vals
    # colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
    # best_rmse = []
    # # Systematically vary the hyperparameter value
    # for curr_val in colsample_bytree_vals:
    #     params['colsample_bytree'] = curr_val
    #
    #     # Perform cross-validation
    #     cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=2,
    #                         num_boost_round=10, early_stopping_rounds=5,
    #                         metrics="rmse", as_pandas=True, seed=123)
    #     # Append the final round rmse to best_rmse
    #     best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])
    # print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)),
    #                    columns=["colsample_bytree", "best_rmse"]))

    # Create the parameter grid: gbm_param_grid
    # gbm_param_grid = {
    #     'scale_pos_weight': [1],
    #     'colsample_bytree': [0.5],
    #     'subsample':[0.9, 1],
    #     'n_estimators': [50],
    #     'reg_alpha': [0.1, 0.3],
    #     'max_depth': [3],
    #     'gamma': [2]
    # }
    #
    # # Instantiate the regressor: gbm
    # gbm = xgb.XGBRegressor()
    #
    # # Perform grid search: grid_mse
    # grid_mse = GridSearchCV(param_grid=gbm_param_grid, estimator=gbm,
    #                         scoring='neg_mean_squared_error', cv=4, verbose=1)
    #
    # # Fit grid_mse to the data
    # grid_mse.fit(X_train, y_train)
    #
    # # Print the best parameters and lowest RMSE
    # print("Best parameters found: ", grid_mse.best_params_)
    # print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))

    # Fit Model
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred = clf.predict(X_test)
    y_pred_train_prob = clf.predict_proba(X_train)
    y_pred_prob = clf.predict_proba(X_test)


    # Model Performance
    # for prob, pred, true in zip(y_pred_prob, y_pred, y_true):
    #     print('(prob, pred, true):', prob, pred, true)
    print('==================Model Statistics==================')
    print('Model Accuracy on Train: ', sklearn.metrics.accuracy_score(y_train, y_pred_train))
    print('Model Accuracy on Test : ', sklearn.metrics.accuracy_score(y_true, y_pred))
    tn_train, fp_train, fn_train, tp_train = metrics.confusion_matrix(y_train, y_pred_train).ravel()
    tn_test, fp_test, fn_test, tp_test = metrics.confusion_matrix(y_true, y_pred).ravel()
    print('Train TPR: ', tp_train/(fp_train+tp_train),'Train TNR: ', tn_train/(fn_train+tn_train))
    print('Test TPR: ', tp_test / (fp_test + tp_test), 'Test TNR: ', tn_test / (fn_test + tn_test))

    # print(metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print(metrics.classification_report(y_true, y_pred, labels=[0, 1]))

    roc_auc = roc_auc_score(np.asarray(y_true), y_pred_prob[:, 1])
    print('AUC: %.3f' % roc_auc)
    fpr, tpr, _ = roc_curve(np.asarray(y_true), y_pred_prob[:, 1])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, marker='.', color=color)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('roc_auc_curve')

    # SDG = SGDClassifier()
    # SDG.fit(X_train, y_train)
    # y2_SDG_model = SDG.predict(X_test)
    # print("SDG Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_SDG_model))
    #
    # KNN = KNeighborsClassifier(n_neighbors=20)
    # KNN.fit(X_train, y_train)
    # y2_KNN_model = KNN.predict(X_test)
    # print("KNN Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_KNN_model))
    return roc_auc


def main(data_path):

    data = pd.read_csv(data_path)
    # print(data.info())
    data['CleanedClaim'] = remove_stopWord(cleaned_claims_lst(data, 'Claim'))
    data['UseNumber'] = data.apply(lambda row: use_numbers(row.Claim), axis=1)
    data['ClaimLength'] = data.apply(lambda row: get_length(row.CleanedClaim), axis=1)


    iter, beat = 0, 0
    while(iter<20):
        data_train, data_test, y_train, y_true = \
            train_test_split(data, data['NoteTrend'], test_size=0.2)
        roc_auc_raw = run_Model(data_train, data_test, y_train, y_true, Optimized=False, color='orange')
        roc_auc_tuned = run_Model(data_train, data_test, y_train, y_true, Optimized=True, color='green')
        if roc_auc_tuned>roc_auc_raw:
            beat += 1
        iter += 1

    print(beat/iter)



    # gnb = GaussianNB()
    # KNN = KNeighborsClassifier(n_neighbors=3)
    # MNB = MultinomialNB()
    # BNB = BernoulliNB()
    # LR = LogisticRegression()
    # SDG = SGDClassifier()
    # LSVC = LinearSVC(dual = False, max_iter=1e6)
    # NSVC = NuSVC()
    # # Train our classifier and test predict
    # KNN.fit(X_train, y_train)
    # y2_KNN_model = KNN.predict(X_test)
    # print("KNN Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_KNN_model))
    #
    # BNB.fit(X_train, y_train)
    # y2_BNB_model = BNB.predict(X_test)
    # print("BNB Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_BNB_model))
    #
    # LR.fit(X_train, y_train)
    # y2_LR_model = LR.predict(X_test)
    # print("LR Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_LR_model))
    #
    # SDG.fit(X_train, y_train)
    # y2_SDG_model = SDG.predict(X_test)
    # print("SDG Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_SDG_model))
    #
    # LSVC.fit(X_train, y_train)
    # y2_LSVC_model = LSVC.predict(X_test)
    # print("LSVC Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_LSVC_model))
    #
    # NSVC.fit(X_train, y_train)
    # y2_NSVC_model = NSVC.predict(X_test)
    # print("NSVC Accuracy :", sklearn.metrics.accuracy_score(y_true, y2_NSVC_model))

if __name__ == '__main__':
    main(data_path='DataClaimTrend.csv')