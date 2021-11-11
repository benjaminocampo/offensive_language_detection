# %%
import pandas as pd
import seaborn as sns
# %%
train_df = pd.read_csv(f"../OLIDv1.0/offenseval_train.csv")
test_df = pd.read_csv(f"../OLIDv1.0/offenseval_test.csv")
# %%
train_df
# %%
sns.histplot(train_df["tweet"].apply(lambda t: len(t)))
# %%
sns.countplot(data=train_df, x="subtask_a")
# %%