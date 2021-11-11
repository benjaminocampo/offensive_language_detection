# %%
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer
from adjustText import adjust_text
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import fasttext
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn as sns
# %%
train_df = pd.read_csv(f"../OLIDv1.0/offenseval_train.csv").drop_duplicates(subset="tweet")
test_df = pd.read_csv(f"../OLIDv1.0/offenseval_test.csv")
# %%
def unsupervised_data_gen(sentences, corpus_file):
    with open(corpus_file, "w") as out:
        for s in sentences:
            out.write(s + "\n")


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
le = LabelEncoder()
le.fit(train_df["subtask_a"])
train_df = train_df.assign(labels=le.transform(train_df["subtask_a"]))
test_df = test_df.assign(labels=le.transform(test_df["subtask_a"]))
# %%
sentences = train_df["tweet"].sample(5000)
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
train_df["vec"] = train_df["tweet"].apply(lambda t: model.get_sentence_vector(t))
test_df["vec"] = train_df["tweet"].apply(lambda t: model.get_sentence_vector(t))
X_train, y_train = np.vstack(train_df["vec"]), train_df["labels"]
X_test, y_test = np.vstack(test_df["vec"]), test_df["labels"]
# %%
kmeans = KMeans(n_clusters=20)
kmeans.fit(X_train)
# %%
train_df = train_df.assign(cluster=kmeans.labels_)
# %%
pd.crosstab(train_df["subtask_a"], train_df["cluster"])
# %%
clf = LogisticRegression(class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
plt.clf()
# %%
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
# %%
sns.histplot(train_df["tweet"].apply(lambda t: len(t)))
# %%
sns.countplot(data=train_df, x="subtask_a")
# %%
bow_vectorizer = CountVectorizer()
bow_vectorizer = bow_vectorizer.fit(train_df["tweet"])
X_train_bow = bow_vectorizer.transform(train_df["tweet"]) 
X_test_bow = bow_vectorizer.transform(test_df["tweet"])
# %%
cluster_df = embedding[embedding["cluster"] == 1]
points = cluster_df.drop(columns=["sentence", "cluster"]).to_numpy()
distances = euclidean_distances(points, kmeans.cluster_centers_[0].reshape(1, -1))
cluster_df = cluster_df.assign(center_distance=distances)
cluster_df = cluster_df.sort_values(by="center_distance")
# %%
cluster_df["center_distance"].head(10)
# %%
for t in cluster_df["sentence"].head(10):
    print(t)
    print()
# %%
