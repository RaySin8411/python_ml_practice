import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram


def make_example():
    variables = ['X', 'Y', 'Z']
    labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
    X = np.random.random_sample([5, 3]) + 10
    df = pd.DataFrame(X, columns=variables, index=labels)
    return df, labels


def cluster_linkage(df, labels):
    row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                            columns=labels, index=labels)
    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    return row_clusters


def plot_dendgram(row_clusters, labels):
    row_dendr = dendrogram(row_clusters, labels=labels)
    plt.tight_layout()
    plt.ylabel('Euclidean distance')
    plt.show()


def plot_another_dengram(df, row_clusters, labels):
    # step1
    fig = plt.figure(figsize=(8, 8), facecolor='white')
    axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    row_dendr = dendrogram(row_clusters, orientation='left')

    # step2
    df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

    # step3
    axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
    cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
    axd.set_xticks([])
    axd.set_yticks([])
    for i in axd.spines.values():
        i.set_visible(False)
    fig.colorbar(cax)
    axm.set_xticklabels([''] + list(df_rowclust.columns))
    axm.set_yticklabels([''] + list(df_rowclust.index))
    plt.show()


def main():
    df, labels = make_example()
    row_clusters = cluster_linkage(df, labels)
    plot_dendgram(row_clusters, labels)
    plot_another_dengram(df, row_clusters, labels)


if __name__ == "__main__":
    main()
