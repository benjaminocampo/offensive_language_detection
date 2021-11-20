# %% [markdown]
# # Integración de clustering para mejorar la clasificación de lenguaje ofensivo
# %%
# Numpy and Pandas
import numpy as np
import pandas as pd

# Fasttext
import fasttext

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer

# Sklearn, XGBoost, and other models
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Utils
from models import NN
from joblib import dump
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
sns.histplot(train_df["tweet"].apply(lambda t: len(t)))
sns.histplot(test_df["tweet"].apply(lambda t: len(t)))
sns.countplot(data=train_df, x="label_name")
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
# - Kmeans con ofensivos y no ofensivos
# - Métricas de pureza
# %%
def plot_elbow(X,
               metric,
               k_range,
               ax,
               title="Elbow Plot",
               title_size=10,
               tick_size=10):
    visualizer = KElbowVisualizer(KMeans(),
                                  k=k_range,
                                  metric=metric,
                                  timings=False,
                                  ax=ax)
    visualizer.fit(X)
    ax.set_title(title, fontsize=title_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

_, axes = plt.subplots(1, 2, figsize=(10, 5))
plot_elbow(X=np.vstack(train_df["vec_off"]),
           k_range=(2, 10),
           metric="distortion",
           title="Elbow with offensive data",
           ax=axes[0])
plot_elbow(X=np.vstack(train_df["vec_mixed"]),
           k_range=(2, 10),
           metric="distortion",
           title="Elbow with mixed data",
           ax=axes[1])
plt.show()
plt.clf()
# %%
kmeans_off = KMeans(n_clusters=4)
kmeans_off.fit(np.vstack(train_df["vec_off"]))

kmeans_mixed = KMeans(n_clusters=5)
kmeans_mixed.fit(np.vstack(train_df["vec_mixed"]))

train_df = train_df.assign(cluster_off=kmeans_off.labels_)
train_df = train_df.assign(cluster_mixed=kmeans_mixed.labels_)

test_df = test_df.assign(
    cluster_off=kmeans_off.predict(np.vstack(test_df["vec_off"])))
test_df = test_df.assign(
    cluster_mixed=kmeans_mixed.predict(np.vstack(test_df["vec_mixed"])))
# %%
pd.crosstab(train_df["label"], train_df["cluster_off"])
# %%
pd.crosstab(train_df["label"], train_df["cluster_mixed"])
# %% [markdown]
# ## Clasificación
# - Fasttext con todos los datos + LGR o XGBoost o NN
# - Fasttext con datos offensivos + LGR o XGBoost o NN
# %%
runs = [
     (LogisticRegression(), {
         "class_weight": ["balanced"],
         "penalty": ["elasticnet"],
         "C": [1, 0.01],
         "solver": ["saga"],
         "max_iter": [1000],
         "l1_ratio": [0, 1, 0.5]
     }),
     (XGBClassifier(), {
         "objective": ["binary:logistic"],
         "n_estimators": [500],
         "gamma": [0.001, 1],
         "max_depth": [20],
         "booster": ["gbtree"],
         "eval_metric": ["logloss"],
         "use_label_encoder": [False],
         "lambda": [0.001, 1]
     }),
    (NN(), {
        "h_size": [32, 64],
        "n_layers": [4, 8],
        "bn_bool": [False],
        "p": [0.2],
        "epochs": [20],
        "batch_size": [32],
        "lr": [0.001],
        "weight_decay": [0]
    }),
]
vec_names = ["vec_off", "vec_mixed"]
experiments = []
for vec_name in vec_names:
    X_train, y_train = np.vstack(
        train_df[vec_name]), train_df["label"].to_numpy()
    X_test, y_test = np.vstack(test_df[vec_name]), test_df["label"].to_numpy()

    results = { }
    for model, grid in runs:
        clf = GridSearchCV(
            estimator=model,
            param_grid=grid,
            scoring="f1",
            refit=True,
            cv=5,
            verbose=3
        )
        clf.fit(X_train, y_train)
        y_pred = clf.best_estimator_.predict(X_test)
        results["model_name"] = model.__class__.__name__
        results["vector"] = vec_name
        results["params"] = clf.best_params_
        results["score"] = f1_score(y_test, y_pred)
        results["pred"] = y_pred
        experiments.append(results)
        dump(clf.best_estimator_, f"{model.__class__.__name__}__{vec_name}.joblib")

experiments_df = pd.DataFrame(experiments)
experiments_df.to_csv("results.csv", index=False)
# %%
experiments_df.assign(cm=experiments_df["pred"].apply(
    lambda pred: confusion_matrix(y_test, pred)))
nof_experiments = len(experiments_df)

best_results_df = experiments_df.iloc[experiments_df["score"].argmax()]

cm = confusion_matrix(y_test, best_results_df["pred"])
cm = cm / np.sum(cm, axis=1).reshape(-1, 1)
sns.heatmap(cm, annot=True)
plt.title(f"{best_results_df['model_name']} {best_results_df['vector']}")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
plt.clf()
# %% [markdown]
# ## Clasificación con etiquetas dada por clustering
# - Solo training set con datos ofensivos
# %%
train_off_df = train_df[train_df["label_name"] == "OFF"]
test_off_df = test_df[test_df["label_name"] == "OFF"]

model = NN()
params = {
    "h_size": 64,
    "n_layers": 8,
    "bn_bool": False,
    "p": 0.2,
    "epochs": 20,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0
}

X_train_off, y_train_off = np.vstack(
    train_off_df["vec_off"]), train_off_df["cluster_off"].to_numpy()
X_test_off, y_test_off = np.vstack(
    test_off_df["vec_off"]), test_off_df["cluster_off"].to_numpy()

model.set_params(**params)
model.fit(X_train_off, y_train_off)
y_pred_off = model.predict(X_test_off)

cm = confusion_matrix(y_test_off, y_pred_off)
cm = cm / np.sum(cm, axis=1).reshape(-1, 1)
sns.heatmap(cm, annot=True)
plt.title(f"NN only_off")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
plt.clf()

dump(model, f"{model.__class__.__name__}__vec_off__off.joblib")
# %%
