# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from functools import reduce
# %%
RAW_DIR = "dataset/raw"
PROCESSED_DIR = "dataset/processed"

labels_A = (
    pd
    .read_csv(f"{RAW_DIR}/labels-levela.csv", header=None)
    .rename(columns={0: "id", 1: "subtask_a"})
)
labels_B = (
    pd
    .read_csv(f"{RAW_DIR}/labels-levelb.csv", header=None)
    .rename(columns={0: "id", 1: "subtask_b"})
)
labels_C = (
    pd
    .read_csv(f"{RAW_DIR}/labels-levelc.csv", header=None)
    .rename(columns={0: "id", 1: "subtask_c"})
)

test_A_df = pd.read_csv(f"{RAW_DIR}/testset-levela.tsv", sep="\t")
test_B_df = pd.read_csv(f"{RAW_DIR}/testset-levelb.tsv", sep="\t")
test_C_df = pd.read_csv(f"{RAW_DIR}/testset-levelc.tsv", sep="\t")

test_A_df = test_A_df.merge(labels_A, on="id")
test_B_df = test_B_df.merge(labels_B, on="id")
test_C_df = test_C_df.merge(labels_C, on="id")
# %%
train_df = pd.read_csv(f"{RAW_DIR}/olid-training-v1.0.tsv", sep="\t")
# %%
test_df = reduce(lambda l, r: pd.merge(l, r, on="id", how="outer"), [
    test_A_df[["id", "tweet", "subtask_a"]], test_B_df[["id", "subtask_b"]],
    test_C_df[["id", "subtask_c"]]
])
# %%
train_df = train_df.drop(columns=["subtask_b", "subtask_c"])
test_df = test_df.drop(columns=["subtask_b", "subtask_c"])
# %%
train_df = train_df.rename(columns={"subtask_a": "label_name"})
test_df = test_df.rename(columns={"subtask_a": "label_name"})
# %%
train_df = train_df.drop_duplicates(subset="tweet")
test_df = test_df.drop_duplicates(subset="tweet")
# %%
le = LabelEncoder()
le.fit(train_df["label_name"])
train_df = train_df.assign(label=le.transform(train_df["label_name"]))
test_df = test_df.assign(label=le.transform(test_df["label_name"]))
# %%
train_df.to_csv(f"{PROCESSED_DIR}/offenseval_train.csv", index=False)
test_df.to_csv(f"{PROCESSED_DIR}/offenseval_test.csv", index=False)
# %%
