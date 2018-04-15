import csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
from scipy import stats

file_name_source = 'program_output/freqmatrix_computer_small.csv'
# file_name_source = 'program_output/freqmatrix_computer.csv'
# file_name_source = 'program_output/freqmatrix_business.csv'
# file_name_source = 'program_output/freqmatrix_mobile.csv'

logistic_files = ["program_output/feature_pvalue_logistic_small.csv", "program_output/original_feature_pvalue_logistic_small.csv",
                  "program_output/p_value_based_model_logistic_small.csv"]

l1_logistic_files = ["program_output/feature_pvalue_small.csv", "program_output/original_feature_pvalue_small.csv",
                  "program_output/p_value_based_model_small.csv"]


def read_file_frequency_matrix(file_name):
    """
    Read the frequency matrix
    :param file_name: name of the file which contains the frequency matrix.
    :return: X - Columns used to predict the data
             Y - The actual class.
    """

    f = open(file_name, encoding="utf-8", errors='ignore')
    reader = csv.reader(f)

    headers = []
    X = []
    Y = []
    for index, row in enumerate(reader):
        if index == 0:
            headers.append(row)
        else:
            Y.append(int(row[-1]))
            X.append(list(map(int, row[1:-1])))

    return X, Y, headers


def logistic_regression(X, Y):
    """
    Compute logistic regression
    :param X: The columns which will be used for analysis.
    :param Y: The expected result of the classification.
    :return: object of classification result and the predicted result.
    """
    logistic = LogisticRegression(C=1e5)  # high c value means regularization effect will be minimal
    logistic.fit(X, Y)
    return logistic


def l1_regularized_logistic_regression(X, Y):
    """
    Compute logistic regression by applying l1 regularization to it.
    :param X: The columns which will be used for analysis.
    :param Y: The expected result of the classification.
    :return: object of classification result and the predicted result.
    """
    logistic = LogisticRegression(penalty='l1')
    logistic.fit(X, Y)
    return logistic


def compute_p_value(headers, X, Y, logistic_output, logistic_files):

    file_name_feature_p_value_filtered = logistic_files[0]
    file_name_feature_p_value = logistic_files[1]
    p_value_based_features = logistic_files[2]

    params = np.append(logistic_output.intercept_, logistic_output.coef_)
    predictions = logistic_output.predict(X)

    newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
    MSE = (sum((Y - predictions) ** 2)) / (len(newX) - len(newX.columns))
    var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    sd_b = np.sqrt(var_b)
    ts_b = params / sd_b

    p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]

    sd_b = np.round(sd_b, 3)
    ts_b = np.round(ts_b, 3)
    p_values = np.round(p_values, 3)
    params = np.round(params, 4)

    myDF3 = pd.DataFrame()
    myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilites"] = [params, sd_b, ts_b,p_values]

    print("Total Features:", len(myDF3["Coefficients"]))

    count = 0
    features_list=[]

    feature_csvlist=[]
    feature_csvlist1=[]

    for index, value in enumerate(myDF3["Probabilites"]):
        list3 = [headers[0][index], value]
        feature_csvlist1.append(list3)

        fw = open(file_name_feature_p_value, 'w', encoding="utf-8", newline='')
        wr = csv.writer(fw)
        wr.writerow(['feature', 'pvalue'])
        for row_value in feature_csvlist1:
            wr.writerow(row_value)

        if value < 0.05:  # and myDF3["Coefficients"][index] >= 0:
            count += 1
            list1 = [index, headers[0][index], value]
            list2 = [headers[0][index], value]
            features_list.append(list1)
            feature_csvlist.append(list2)

    print(features_list)
    print("Shortlisted: ",  len(features_list))
    fw = open(file_name_feature_p_value_filtered, 'w', encoding="utf-8", newline='')
    wr = csv.writer(fw)
    wr.writerow(['feature', 'pvalue'])
    for row_value in feature_csvlist:
        wr.writerow(row_value)

    positives = []
    negatives = []

    for row in features_list:
        index = row[0] - 1
        for row_number, count in enumerate(i for i in X):
            if count[index] > 0:

                counter = 0
                if Y[row_number] == 0:
                    while counter < count[index]:
                        positives.append(headers[0][index + 1])
                        counter += 1
                else:
                    while counter < count[index]:
                        negatives.append(headers[0][index + 1])
                        counter += 1

    distinct_positives = list(set(positives))
    distinct_negatives = list(set(negatives))

    print('positive reviews', len(distinct_positives), ", list:", distinct_positives)
    print('negative reviews', len(distinct_negatives), ", list:", distinct_negatives)

    outputlist = []
    for feature in features_list:
        row = []
        word, p_val = feature[1], feature[2]
        row.append(word)

        row.append(positives.count(word))
        row.append(negatives.count(word))

        row.append(p_val)
        outputlist.append(row)

    fw = open(p_value_based_features, 'w', encoding="utf-8", newline='')
    wr = csv.writer(fw)
    wr.writerow(['Feature', 'Positive(0)', 'Negative(1)', 'p-value'])
    for row in outputlist:
        wr.writerow(row)


def main():
    np.seterr(all='ignore')  # Suppress numpy Warnings.
    X, Y, headers = read_file_frequency_matrix(file_name_source)
    X_train = X[0:600]
    Y_train = Y[0:600]
    logistic_regression_results = logistic_regression(X_train, Y_train)
    l1_logistic_regression_results = l1_regularized_logistic_regression(X_train, Y_train)

    X_test = X[600::]
    Y_test = Y[600::]

    print("********************** Logistic Regression **********************")
    Y_predicted = logistic_regression_results.predict(X_test)
    print("Report: \n", metrics.classification_report(Y_test, Y_predicted))
    print("Accuracy: ", metrics.accuracy_score(Y_test, Y_predicted))

    print("********************** After l1 Regularization **********************")
    Y_predicted = l1_logistic_regression_results.predict(X_test)
    print("Report: \n", metrics.classification_report(Y_test, Y_predicted))
    print("Accuracy: ", metrics.accuracy_score(Y_test, Y_predicted))

    print("\n********************** Logistic Regression **********************")
    logistic_regression_results = logistic_regression(X, Y)
    compute_p_value(headers, X, Y, logistic_regression_results, logistic_files)
    print("********************** After l1 Regularization **********************")
    l1_logistic_regression_results = l1_regularized_logistic_regression(X, Y)
    compute_p_value(headers, X, Y, l1_logistic_regression_results, l1_logistic_files)

if __name__ == '__main__':
    main()
