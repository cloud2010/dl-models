from sklearn.datasets import load_svmlight_file
# from sklearn.cross_validation import *
from sklearn.model_selection import *
from sklearn import metrics
# import numpy as np
import sys
# import os
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib


# from sklearn.linear_model import LogisticRegression
# import string


class GbdtParamter:
    _nams = ["loss", "learning_rate", "n_estimators",
             "max_depth", "subsample", "cv", "p"]

    def __init__(self, option=None):
        if option is None:
            option = ''
        self.parse_option(option)

    def parse_option(self, option):
        if isinstance(option, list):
            argv = option
        elif isinstance(option, str):
            argv = option.split()
        else:
            raise TypeError("arg 1 should be a list or a str .")
        self.set_to_default_values()

        i = 0
        while i < len(argv):
            if argv[i] == "-ls":
                i = i + 1
                self.loss = argv[i]
            elif argv[i] == "-lr":
                i = i + 1
                self.learning_rate = float(argv[i])
            elif argv[i] == "-ns":
                i = i + 1
                self.n_estimators = int(argv[i])
            elif argv[i] == "-md":
                i = i + 1
                self.max_depth = int(argv[i])
            elif argv[i] == "-sub":
                i = i + 1
                self.subsample = float(argv[i])
            elif argv[i] == "-cv":
                i = i + 1
                self.cv = int(argv[i])
            elif argv[i] == "-p":
                i = i + 1
                self.p = argv[i]
            else:
                raise ValueError(
                    "Wrong options.Only -ls(loss) -lr(learning_rate) -ns(n_estimators) -md(max_depth) -sub("
                    "subssample),-cv,-p(testFile)")
            i += 1

    def set_to_default_values(self):
        self.loss = "deviance"
        self.learning_rate = 0.1
        self.n_estimators = 100
        self.max_depth = 3
        self.subsample = 1
        self.cv = 0
        self.p = ""


def read_data(data_file):
    try:
        t_X, t_y = load_svmlight_file(data_file)
        return t_X, t_y
    except ValueError as e:
        print(e)


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y, para):
    model = GradientBoostingClassifier(n_estimators=para.n_estimators, max_depth=para.max_depth,
                                       learning_rate=para.learning_rate, loss=para.loss,
                                       subsample=para.subsample)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    def exit_with_help():
        print(
            "Usage: gbdt.py  [-ls (loss: deviance,exponential),-lr(learning_rate 0.1),-ns(n_estimators 100),"
            "-md(max_depth 3),-sub(subsample 1),-cv (10),-p testFile] dataset")
        sys.exit(1)

    if len(sys.argv) < 2:
        exit_with_help()

    # time statistics
    start_time = time.clock()
    print("\nTime start: " + time.strftime('%Y-%m-%d %H:%M:%S',
                                           time.localtime(time.time())))

    dataset_path_name = sys.argv[-1]
    option = sys.argv[1:-1]
    try:
        train_X, train_Y = read_data(dataset_path_name)
        train_X = train_X.todense()
        para = GbdtParamter(option)
        gbdt = gradient_boosting_classifier(train_X, train_Y, para)

        # Model persistence
        joblib.dump(gbdt,
                    '{0}-cv-{1}-ns-{2}-md-{3}.pkl'.format(str(dataset_path_name), str(para.cv), str(para.n_estimators),
                                                          str(para.max_depth)))

        if para.cv > 0:
            accuracy = cross_val_score(
                gbdt, train_X, train_Y, cv=10, scoring='accuracy')
            roc = cross_val_score(gbdt, train_X, train_Y,
                                  cv=10, scoring='roc_auc')
            # console output redirect to log file
            out = open(
                '{0}-cv-{1}-ns-{2}-md-{3}.log'.format(str(dataset_path_name), str(para.cv), str(para.n_estimators),
                                                      str(para.max_depth)), 'w')
            print("\nParameters:")
            out.write("\nParameters:")

            print('\nn_estimators:{0} max_depth:{1} learning_rate:{2} loss:{3} subsample:{4}'.format(
                str(para.n_estimators), str(para.max_depth), str(
                    para.learning_rate), str(para.loss),
                str(para.subsample)))
            out.write('\nn_estimators:{0} max_depth:{1} learning_rate:{2} loss:{3} subsample:{4}'.format(
                str(para.n_estimators), str(para.max_depth), str(
                    para.learning_rate), str(para.loss),
                str(para.subsample)))

            print("\n10 cross validation result:\n")
            out.write("\n10 cross validation result:\n")

            print("Accuracy:" + str(accuracy.mean()) + "\n")
            out.write("Accuracy:" + str(accuracy.mean()) + "\n")

            print("Receiver Operating Characteristic(AUC):" +
                  str(roc.mean()) + "\n")
            out.write("Receiver Operating Characteristic(AUC):" +
                      str(roc.mean()) + "\n")
            predicted = cross_val_predict(gbdt, train_X, train_Y, cv=10)

            print("Confusion Matrix:" + "\n")
            out.write("Confusion Matrix:" + "\n")

            print(metrics.confusion_matrix(train_Y, predicted))
            out.write(str(metrics.confusion_matrix(train_Y, predicted)))

            print(
                "\nThe feature importances (the higher, the more important the feature)" + "\n")
            out.write(
                "\nThe feature importances (the higher, the more important the feature)" + "\n")

            print(gbdt.feature_importances_)
            out.write(str(gbdt.feature_importances_))

            # end time
            end_time = time.clock()
            print("\nTime consumption: %f s" % (end_time - start_time))
            out.write("\nTime consumption: %f s" % (end_time - start_time))
            # file stream close
            out.close()

        if para.p != "":
            test_x, test_y = read_data(para.p)
            predict = gbdt.predict(test_x.todense())
            prob = gbdt.predict_proba(test_x.todense())
            out = open('predict', 'w')
            out.write("origin\tpredict\tprob\n")
            for i in range(predict.shape[0]):
                if i % 1000 == 0:
                    print("instance:" + str(i))
                out.write(str(test_y[i]) + "\t" +
                          str(predict[i]) + "\t" + str(prob[i]) + '\n')
            out.close()
    except(IOError, ValueError) as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)
