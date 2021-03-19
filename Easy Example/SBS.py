from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores, subsets = [], []
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


def main():
    import pandas as pd
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                          header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3, random_state=0, stratify=y)
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)

    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=1)

    sbs.fit(X_train_std, y_train)

    k_feat = [len(k) for k in sbs.subsets_]
    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylim([0.7, 1.02])
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()

    # k3, 3 features
    k3 = list(sbs.subsets_[10])
    print(df_wine.columns[1:][k3])

    knn.fit(X_train_std, y_train)
    print('Training accuracy', 100 * round(knn.score(X_train_std, y_train), 4))
    print('Test accuracy', 100 * round(knn.score(X_test_std, y_test), 4))

    knn.fit(X_train_std[:, k3], y_train)
    print('Training accuracy', 100 * round(knn.score(X_train_std[:, k3], y_train), 4))
    print('Test accuracy', 100 * round(knn.score(X_test_std[:, k3], y_test), 4))

    def plot_decision_regions(X, y, classifier, resolution=0.02):
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt

        # setup marker generator and color map
        markers, colors = ('s', 'x', 'o', '^', 'v'), ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.6, c=cmap(idx), edgecolor='black',
                        marker=markers[idx], label=cl)

    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=2)
    sbs.fit(X_train_std, y_train)
    lr = LogisticRegression()
    X_train_sbs = sbs.transform(X_train_std)
    lr.fit(X_train_sbs, y_train)
    plot_decision_regions(X_train_sbs, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    def plot_svc_decision_function(model, ax=None, plot_support=True):
        """Plot the decision function for a two-dimension SVC"""
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 建立grid以評估模型
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        # 繪出決策邊界
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # 繪出支持向量
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=1, facecolors='none')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    lr = LogisticRegression()
    X_train_pca = pca.fit_transform(X_train_std)
    lr.fit(X_train_pca, y_train)
    plot_decision_regions(X_train_pca, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    knn = KNeighborsClassifier(n_neighbors=5)
    sbs = SBS(knn, k_features=2)
    sbs.fit(X_train_std, y_train)
    lr = LogisticRegression()
    X_train_sbs = sbs.transform(X_train_std)
    lr.fit(X_train_sbs, y_train)
    plot_decision_regions(X_train_sbs, y_train, classifier=lr)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend(loc='lower left')
    plt.show()

    def plot_svc_decision_function(model, ax=None, plot_support=True):
        """Plot the decision function for a two-dimension SVC"""
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 建立grid以評估模型
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)

        # 繪出決策邊界
        ax.contour(X, Y, P, colors='k',
                   levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])

        # 繪出支持向量
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=300, linewidth=1, facecolors='none')

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        from sklearn.svm import SVC  # "Support vector classifier"

        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_std)
        model = SVC(kernel='linear', C=1E10)
        model.fit(X_train_pca, y_train)
        plot_decision_regions(X_train_pca, y_train, classifier=model)

        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='lower left')
        plt.show()

        knn = KNeighborsClassifier(n_neighbors=5)
        sbs = SBS(knn, k_features=2)
        sbs.fit(X_train_std, y_train)
        X_train_sbs = sbs.transform(X_train_std)
        model = SVC(kernel='linear', C=1E10)
        model.fit(X_train_sbs, y_train)
        plot_decision_regions(X_train_sbs, y_train, classifier=model)

        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='lower left')
        plt.show()


if __name__ == '__main__':
    main()
