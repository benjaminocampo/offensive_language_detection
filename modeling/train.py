# %%
import pandas as pd
import fasttext
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

DATA_DIR = "OLIDv1.0"
# %%
def unsupervised_data_gen(sentences, corpus_file):
    with open(corpus_file, "w") as out:
        for s in sentences:
            out.write(s + "\n")


def get_word_embedding(vocab, vectorize):
    return (
        pd.DataFrame({word: vectorize(word) for word in vocab})
        .transpose()
        .reset_index()
        .rename(columns={"index": "word"})
    )

def get_word_embedding_2DTSNE(vocab, model):
    embedding = get_word_embedding(vocab, model)
    X = embedding.drop(columns=["word"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(vocab, columns=["word"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE

def get_sentence_embedding(sentences, vectorize):
    return (
        pd.DataFrame({s: vectorize(s) for s in sentences})
        .transpose()
        .reset_index()
        .rename(columns={"index": "sentence"})
    )

def get_sentence_embedding_2DTSNE(sentences, vectorize):
    embedding = get_sentence_embedding(sentences, vectorize)
    X = embedding.drop(columns=["sentence"])
    X_TSNE = TSNE(n_components=2).fit_transform(X)
    embedding_TSNE = pd.concat(
        [pd.DataFrame(sentences, columns=["sentence"]),
         pd.DataFrame(X_TSNE)], axis=1)
    return embedding_TSNE


def plot_2DTSNE(X_TSNE,
                expressions,
                ax,
                point_size=50,
                legend_size=20,
                tick_size=20,
                annotation_size=20,
                annotate=True):
    nof_words = len(X_TSNE)
    ax.scatter(X_TSNE[:, 0],
               X_TSNE[:, 1],
               s=point_size)
    ax.set_title("t-SNE Plot", fontsize=legend_size)
    annot_list = []

    if annotate:
        nof_words_to_annotate = 20
        for i in np.random.randint(nof_words, size=nof_words_to_annotate):
            a = ax.annotate(expressions[i], (X_TSNE[i, 0], X_TSNE[i, 1]),
                            size=annotation_size)
            annot_list.append(a)
        adjust_text(annot_list)

    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_2D_kmeans(X_TSNE,
                   km_model,
                   ax,
                   marker_size=2,
                   legend_size=20,
                   tick_size=20):
    cluster_df = pd.DataFrame(X_TSNE)
    cluster_df = cluster_df.assign(label=km_model.labels_)
    sns.scatterplot(data=cluster_df, x=0, y=1, hue="label", palette="tab20", ax=ax)
    ax.legend(bbox_to_anchor=(1.02, 1.02),
              loc='upper left',
              markerscale=marker_size,
              prop={"size": legend_size})
    ax.set_title("KMeans Plot", fontsize=legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)


def plot_silhouette(X, km_model, ax, title_size=20, tick_size=20):
    visualizer = SilhouetteVisualizer(km_model, colors='tab20', ax=ax)
    visualizer.fit(X)
    ax.set_title("Silhoutte Plot", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

def plot_elbow(X, estimator, metric, k_range, ax, title_size=10, tick_size=10):
    visualizer = KElbowVisualizer(estimator(),
                                  k=k_range,
                                  metric=metric,
                                  timings=False,
                                  ax=ax)
    visualizer.fit(X)
    ax.set_title("Elbow Plot", fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
# %%
train_df = pd.read_csv(f"../{DATA_DIR}/offenseval_train.csv")
test_df = pd.read_csv(f"../{DATA_DIR}/offenseval_test.csv")
sentences = train_df["cleaned_tweet"].sample(5000).drop_duplicates().tolist()
corpus_file = "tweets.txt"
# %%
unsupervised_data_gen(sentences, corpus_file)
# %%
model = fasttext.train_unsupervised(corpus_file,
                                    model="cbow",
                                    lr=0.3,
                                    epoch=100,
                                    dim=100,
                                    wordNgrams=4,
                                    ws=4)

# %%
model.get_nearest_neighbors("shits")
# %%
embedding = get_sentence_embedding(sentences, model.get_sentence_vector)
embedding_TSNE = get_sentence_embedding_2DTSNE(sentences,
                                               model.get_sentence_vector)
X = embedding.drop(columns=["sentence"]).to_numpy()
X_TSNE = embedding_TSNE.drop(columns=["sentence"]).to_numpy()
# %%
fig, ax_elbow = plt.subplots()
k_range = (2, 20)
plot_elbow(X, KMeans, "distortion", k_range, ax_elbow)
plt.show()
# %%
kmeans = KMeans(n_clusters=8)
kmeans.fit(X)
# %%
_, (ax_silhouette, ax_kmeans) = plt.subplots(1, 2, figsize=(30, 10))
plot_silhouette(X, kmeans, ax_silhouette)
plot_2D_kmeans(X_TSNE, kmeans, ax_kmeans)
ax_silhouette.grid()
ax_kmeans.grid()
plt.show()
# %%
embedding = embedding.assign(cluster=kmeans.labels_)
# %%