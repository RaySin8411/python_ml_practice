import six

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):

        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax
        # call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        # Predict class labels for X.

        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)

        else:
            # 'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                                           axis=1, arr=predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        # Predict class probabilities for X.
        probs = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probs, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


def main_iris():
    ## Deal with iris data
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    iris = datasets.load_iris()
    X, y = iris.data[50:, [1, 2]], iris.target[50:]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)

    ## Bulid Model: LogisticRegression, DecsisionTreeClassifier, KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    clf1 = LogisticRegression(penalty='l2', C=0.0001, random_state=1)
    clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
    clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')
    pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
    pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])
    clf_labels = ['Logistic regression', 'Decision tree', 'KNN']

    ## Training Results
    mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    clf_labels += ['Majority voting']
    all_clf = [pipe1, clf2, pipe3, mv_clf]
    print('10-fold cross validation:\n')
    for clf, label in zip(all_clf, clf_labels):
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train,
                                 cv=10, scoring='roc_auc')
        print('Accuracy: %0.2f (+/- %0.2f) [%s]' % (scores.mean(), scores.std(), label))

    ## plot
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt
    colors = ['black', 'orange', 'blue', 'green']
    linestyles = [':', "--", '-.', '-']
    for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
        # assuming the label of the positive class is 1
        y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
        roc_auc = auc(x=fpr, y=tpr)
        plt.plot(fpr, tpr, color=clr, linestyle=ls,
                 label='%s (auc = %0.2f)' % (label, roc_auc))
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.grid(alpha=0.5)
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
    plt.show()

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    from itertools import product
    x_min = X_train_std[:, 0].min() - 1
    x_max = X_train_std[:, 0].max() + 1
    y_min = X_train_std[:, 1].min() - 1
    y_max = X_train_std[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=2, ncols=2, sharex='col',
                            sharey='row', figsize=(7, 5))
    for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
        clf.fit(X_train_std, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                      X_train_std[y_train == 0, 1],
                                      c='blue', marker='^', s=50)
        axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                      X_train_std[y_train == 1, 1],
                                      c='green', marker='o', s=50)
        axarr[idx[0], idx[1]].set_title(tt)

    plt.text(-3.5, -4.5, s='Sepal width [standardized]',
             ha='center', va='center', fontsize=12)
    plt.text(-12.5, 4.5, s='Petal length [standardized]',
             ha='center', va='center', fontsize=12, rotation=90)
    plt.show()


def main_wine():
    import pandas as pd
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                       'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    # drop 1 class
    df_wine = df_wine[df_wine['Class label'] != 1]
    y = df_wine['Class label'].values
    X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',
                                  random_state=1, max_depth=None)
    bag = BaggingClassifier(base_estimator=tree,
                            n_estimators=500, max_samples=1.0, max_features=1.0,
                            bootstrap=True, bootstrap_features=False, n_jobs=1,
                            random_state=1)

    from sklearn.metrics import accuracy_score
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f' % (tree_train, tree_test))

    bag = bag.fit(X_train, y_train)
    y_train_pred = bag.predict(X_train)
    y_test_pred = bag.predict(X_test)
    bag_train = accuracy_score(y_train, y_train_pred)
    bag_test = accuracy_score(y_test, y_test_pred)
    print('Bagging train/test accuracies %.3f/%.3f' % (bag_train, bag_test))

    import matplotlib.pyplot as plt
    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))
    for idx, clf, tt in zip([0, 1], [tree, bag], ['Decision tree', 'Bagging']):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green',
                           marker='o')
        axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(10.2, -1.2, s='OD280/OD315 of diluted wines',
             ha='center', va='center', fontsize=12)
    plt.show()

    tree = DecisionTreeClassifier(criterion='entropy',
                                  random_state=1, max_depth=1)
    tree = tree.fit(X_train, y_train)
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    tree_train = accuracy_score(y_train, y_train_pred)
    tree_test = accuracy_score(y_test, y_test_pred)
    print('Decision tree train/test accuracies %.3f/%.3f'
          % (tree_train, tree_test))

    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(base_estimator=tree,
                             n_estimators=500, learning_rate=0.1, random_state=1)
    ada = ada.fit(X_train, y_train)
    y_train_pred = ada.predict(X_train)
    y_test_pred = ada.predict(X_test)
    ada_train = accuracy_score(y_train, y_train_pred)
    ada_test = accuracy_score(y_test, y_test_pred)
    print('AdaBoost train/test accuracies %.3f/%.3f'
          % (ada_train, ada_test))

    x_min = X_train[:, 0].min() - 1
    x_max = X_train[:, 0].max() + 1
    y_min = X_train[:, 1].min() - 1
    y_max = X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(nrows=1, ncols=2, sharex='col', sharey='row', figsize=(8, 3))
    for idx, clf, tt in zip([0, 1], [tree, ada], ['Decision tree', 'AdaBoost']):
        clf.fit(X_train, y_train)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axarr[idx].contourf(xx, yy, Z, alpha=0.3)
        axarr[idx].scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                           c='blue', marker='^')
        axarr[idx].scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='green',
                           marker='o')
        axarr[idx].set_title(tt)
    axarr[0].set_ylabel('Alcohol', fontsize=12)
    plt.text(10.2, -1.2, s='OD280/OD315 of diluted wines',
             ha='center', va='center', fontsize=12)
    plt.show()




if __name__ == '__main__':
    main_iris()
    main_wine()
