import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

plt.rcParams['axes.unicode_minus'] = False


def make_blobs_example():
    X, y = make_blobs(n_samples=150, n_features=2,
                      centers=3, cluster_std=0.5, shuffle=True, random_state=0)
    return X, y


def make_moons_example():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    return X, y


def plot_example(X):
    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolor='black', s=50)
    plt.grid()
    plt.show()


def plot_kmeans(X):
    km = KMeans(n_clusters=3, init='random', n_init=10,
                max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightgreen', marker='s',
                edgecolor='black', s=50, label='cluster 1')

    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='orange', marker='o',
                edgecolor='black', s=50, label='cluster 2')

    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], c='lightblue', marker='v',
                edgecolor='black', s=50, label='cluster 3')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', edgecolor='black', label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()


def elbow_method(X):
    distortions = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10,
                    max_iter=300, random_state=0)
        km.fit(X)
        distortions.append(km.inertia_)
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()
    return distortions


def plot_centroids(X):
    km = KMeans(n_clusters=2, init='random', n_init=10,
                max_iter=300, tol=1e-04, random_state=0)
    y_km = km.fit_predict(X)
    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightgreen', marker='s',
                edgecolor='black', s=50, label='cluster 1')

    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='orange', marker='o',
                edgecolor='black', s=50, label='cluster 2')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                s=250, marker='*', c='red', edgecolor='black', label='centroids')
    plt.legend()
    plt.grid()
    plt.show()


def compare_metod(X):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))

    km = KMeans(n_clusters=2, random_state=0)
    y_km = km.fit_predict(X)
    ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightblue', marker='o',
                edgecolor='black', s=40, label='cluster 1')
    ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='red', marker='s',
                edgecolor='black', s=40, label='cluster 2')
    ax1.set_title('K-means clustering')

    ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                 linkage='complete')
    y_ac = ac.fit_predict(X)
    ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue', marker='o',
                edgecolor='black', s=40, label='cluster 1')
    ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c='red', marker='s',
                edgecolor='black', s=40, label='cluster 2')
    ax2.set_title('Agglomerative clustering')

    db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
    y_db = db.fit_predict(X)
    ax3.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c='lightblue', marker='o',
                edgecolor='black', s=40, label='cluster 1')
    ax3.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c='red', marker='s',
                edgecolor='black', s=40, label='cluster 2')
    ax3.set_title('DBSCAN')
    plt.tight_layout()
    plt.legend()
    plt.show()


def main():
    X, y = make_blobs_example()
    plot_example(X)
    plot_kmeans(X)
    distortions = elbow_method(X)
    plot_centroids(X)

    X, y = make_moons_example()
    plot_example(X)
    compare_metod(X)


if __name__ == "__main__":
    main()
