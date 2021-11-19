# %% [markdown]
# # Integración de clustering para mejorar la clasificación de lenguaje ofensivo
# %% [markdown]
# ## Descripción de la tarea
# TODO: Mencionar los experimentos y armar indices.
# %%
# Numpy and Pandas
import numpy as np
import pandas as pd

# Fasttext
import fasttext

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

# Sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# %% [markdown]
# ## Exploración de datos
# %%
train_df = pd.read_csv(f"dataset/processed/offenseval_train.csv")
test_df = pd.read_csv(f"dataset/processed/offenseval_test.csv")
# %%
train_df
# %%
test_df
# %%
# %% [markdown]
# ## Codificación de tweets
# %%
def unsupervised_data_gen(sentences, corpus_file):
    with open(corpus_file, "w") as out:
        for s in sentences:
            out.write(s + "\n")

off_sentences = train_df.loc[train_df["label_name"] == "OFF", "tweet"]
mixed_sentences = train_df["tweet"].sample(5000)

unsupervised_data_gen(off_sentences, "offensive_sentences.txt")
unsupervised_data_gen(mixed_sentences, "mixed_sentences.txt")

model_off = fasttext.train_unsupervised("offensive_sentences.txt",
                                        model="cbow",
                                        lr=0.3,
                                        epoch=100,
                                        dim=100,
                                        wordNgrams=4,
                                        ws=4)
model_mixed = fasttext.train_unsupervised("mixed_sentences.txt",
                                          model="cbow",
                                          lr=0.3,
                                          epoch=100,
                                          dim=100,
                                          wordNgrams=4,
                                          ws=4)

train_df = train_df.assign(
    vec_off=train_df["tweet"].apply(lambda t: model_off.get_sentence_vector(t)),
    vec_mixed=train_df["tweet"].apply(lambda t: model_mixed.get_sentence_vector(t))
)
test_df = test_df.assign(
    vec_off=test_df["tweet"].apply(lambda t: model_off.get_sentence_vector(t)),
    vec_mixed=test_df["tweet"].apply(lambda t: model_mixed.get_sentence_vector(t))
)
# %% [markdown]
# ## Clustering
# TODO:
# - Kmeans con ofensivos y no ofensivos
# - Gráficos de silhouette
# - Métricas de pureza
# - Que hay en los clusters
# %% [markdown]
# ## Clasificación
# TODO:
# - Fasttext con todos los datos + LGR o XGBoost o NN
# - Fasttext con datos offensivos + LGR o XGBoost o NN
# - Lo mismo pero con undersampling
# %% [markdown]
# ## Clasificación con etiquetas dada por clustering
# TODO:
# - Solo training set con datos ofensivos
# %%


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
model = fasttext.train_unsupervised(corpus_file,
                                    model="cbow",
                                    lr=0.3,
                                    epoch=100,
                                    dim=100,
                                    wordNgrams=4,
                                    ws=4)
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

print(classification_report(y_test, y_pred))
# %%
sns.histplot(train_df["tweet"].apply(lambda t: len(t)))
# %%
sns.countplot(data=train_df, x="subtask_a")
# %%
train_off_df = train_df[train_df["subtask_a"] == "OFF"]
train_not_df = train_df[train_df["subtask_a"] == "NOT"]
# %%
sentences = train_off_df["tweet"]
corpus_file = "tweets.txt"
model = fasttext.train_unsupervised(corpus_file,
                                    model="cbow",
                                    lr=0.3,
                                    epoch=100,
                                    dim=100,
                                    wordNgrams=4,
                                    ws=4)
# %%
train_off_df = train_off_df.assign(vec_off=train_off_df["tweet"].apply(
    lambda t: model.get_sentence_vector(t)))
train_not_df = train_not_df.assign(vec_off=train_not_df["tweet"].apply(
    lambda t: model.get_sentence_vector(t)))
train_new_df = pd.concat([train_off_df, train_not_df])
X_train_new, y_train_new = np.vstack(
    train_new_df["vec_off"]), train_new_df["labels"]

test_off_df = test_df[test_df["subtask_a"] == "OFF"]
test_not_df = test_df[test_df["subtask_a"] == "NOT"]

test_off_df = test_off_df.assign(
    vec_off=test_off_df["tweet"].apply(lambda t: model.get_sentence_vector(t)))
test_not_df = test_not_df.assign(
    vec_off=test_not_df["tweet"].apply(lambda t: model.get_sentence_vector(t)))
test_new_df = pd.concat([test_off_df, test_not_df])
X_test_new, y_test_new = np.vstack(test_new_df["vec_off"]), test_new_df["labels"]

clf = SVC(class_weight="balanced")
clf.fit(X_train_new, y_train_new)

y_pred = clf.predict(X_test_new)
print(classification_report(y_test_new, y_pred))
# %%
cm = confusion_matrix(y_test_new, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()
plt.clf()
# %%
test_off_df = test_off_df.assign(
    new_label=kmeans_off.predict(np.vstack(test_off_df["vec_off"])))
# %%
test_off_df
# %%
X_test_off, y_test_off = np.vstack(test_off_df["vec_off"]), test_off_df["new_label"]
# %%
y_pred_off = clf.predict(X_test_off)
# %%
print(classification_report(y_test_off, y_pred_off))
# %%
test_off_df[test_off_df["new_label"] == 0]
# %%
test_off_df[test_off_df["new_label"] == 1]
# %%
train_not_df = train_not_df.assign(
    new_label=10)
# %%
