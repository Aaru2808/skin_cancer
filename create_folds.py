import os
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":  # Fixed the condition here
    df = pd.read_csv("train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe
    y = df.target.values  # Assuming 'target' is the column to predict

    kf = model_selection.StratifiedKFold(n_splits=5)  # Using 1 split, adjust as needed
    for fold_, (_, _) in enumerate(kf.split(X=df, y=y)):
        df.loc[:, "kfold"] = fold_

    df.to_csv(("train_folds.csv"), index=False)
