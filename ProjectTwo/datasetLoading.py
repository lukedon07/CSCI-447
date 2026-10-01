import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, name, task, X, y, is_numeric, names, class_names=None):
        self.name, self.task = name, task
        self.X, self.y = X, y
        self.is_numeric, self.names = is_numeric, names
        self.class_names = class_names

    def __len__(self):
        return len(self.y)

    def subset(self, idx):
        return [c[idx] for c in self.X], self.y[idx]


def _build(name, task, df, target, cat_cols, num_cols, target_transform=None):
    X = [df[c].to_numpy(dtype=float) for c in num_cols] + \
        [df[c].astype(str).to_numpy(dtype=object) for c in cat_cols]
    is_num = [True] * len(num_cols) + [False] * len(cat_cols)
    names = list(num_cols) + list(cat_cols)
    if task == "classification":
        classes, y = np.unique(df[target].astype(str), return_inverse=True)
        return Dataset(name, task, X, y.astype(int), is_num, names, list(classes))
    y = df[target].to_numpy(dtype=float)
    if target_transform:
        y = target_transform(y)
    return Dataset(name, task, X, y, is_num, names)


def load_dataset(name, d):
    p = lambda f: os.path.join(d, f)
    if name == "breast":
        cols = ["id", "clump", "cell_size", "cell_shape", "adhesion", "epi_size",
                "bare_nuclei", "bland_chromatin", "norm_nucleoli", "mitoses", "class"]
        df = pd.read_csv(p("breast-cancer-wisconsin.data"), header=None,
                         names=cols, dtype=str).drop(columns="id")  # unique id
        mode = df.loc[df["bare_nuclei"] != "?", "bare_nuclei"].mode()[0]
        df["bare_nuclei"] = df["bare_nuclei"].replace("?", mode)  # impute missing
        # 1-10 ordinal scores -> categorical (multiway splits), per assignment
        return _build("breast", "classification", df, "class", cols[1:-1], [])
    if name == "car":
        cols = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        df = pd.read_csv(p("car.data"), header=None, names=cols, dtype=str)
        return _build("car", "classification", df, "class", cols[:-1], [])
    if name == "vote":
        cols = ["party"] + [f"v{i}" for i in range(1, 17)]
        df = pd.read_csv(p("house-votes-84.data"), header=None, names=cols, dtype=str)
        # '?' means "did not vote yes/no"; kept as its own categorical value
        return _build("vote", "classification", df, "party", cols[1:], [])
    if name == "abalone":
        cols = ["sex", "length", "diameter", "height", "whole", "shucked",
                "viscera", "shell", "rings"]
        df = pd.read_csv(p("abalone.data"), header=None, names=cols)
        return _build("abalone", "regression", df, "rings", ["sex"], cols[1:-1])
    if name == "machine":
        cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH",
                "CHMIN", "CHMAX", "PRP", "ERP"]
        df = pd.read_csv(p("machine.data"), header=None, names=cols)
        df = df.drop(columns=["model", "ERP"])  # unique id; authors' own estimate
        return _build("machine", "regression", df, "PRP", ["vendor"],
                      ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"])
    if name == "forest":
        df = pd.read_csv(p("forestfires.csv"))
        num = ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
        # area is heavily skewed with many zeros -> ln(area + 1), as the data's authors suggest
        return _build("forest", "regression", df, "area", ["month", "day"], num,
                      target_transform=np.log1p)
    raise ValueError(name)


ALL_DATASETS = ["breast", "car", "vote", "abalone", "machine", "forest"]