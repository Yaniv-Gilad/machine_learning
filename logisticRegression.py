import numpy as np
import pandas as pd  # tables

# sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing  # normalize


def calc_fpr_tpr(mat, dig1, dig2):
    fpr = (mat[dig1][dig2]) / (mat[dig1][dig2] + mat[dig2][dig2])
    tpr = (mat[dig1][dig1]) / (mat[dig1][dig1] + mat[dig2][dig1])
    return fpr, tpr


if __name__ == '__main__':
    data, y = load_digits(return_X_y=True, n_class=10)

    # select only 8 and 9 digits
    dig1 = 8
    dig2 = 9
    testSize = 0.3
    data = data[np.logical_or(y == dig1, y == dig2)]
    y = y[np.logical_or(y == dig1, y == dig2)]

    # normal the data
    scaler = preprocessing.StandardScaler()
    d = scaler.fit_transform(data)
    data_scaled_df = pd.DataFrame(d)

    x_train, x_test, y_train, y_test = train_test_split(d, y, test_size=testSize, random_state=0)
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # print confusion matrics
    y_pred = model.predict(x_test)
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    # print stat
    fpr, tpr = calc_fpr_tpr(confusion_matrix, dig1, dig2)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('Precision: ',
          ((confusion_matrix[dig1][dig1]) / (confusion_matrix[dig1][dig1] + confusion_matrix[dig1][dig2])))
    print("FPR: {}\nTPR: {}\n".format(fpr, tpr))

    # ROC curve
    y_pred_proba = model.predict_proba(x_test)[::, 1]  # return probability
    ones_zeros = y_test
    for index, number in enumerate(y_test):
        if number == dig1:
            ones_zeros[index] = 0
        else:
            ones_zeros[index] = 1

    fpr, tpr, threshold = metrics.roc_curve(ones_zeros, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.title(f'ROC curve with AUC: {auc}')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'r--', label='y = x')
    plt.plot(0, 0, 'go', label='S = 0')
    plt.plot(1, 1, 'ko', label='S = 1')

    plt.legend(loc=4)
    plt.show()
    print(f'ROC curve with AUC: {auc}')
