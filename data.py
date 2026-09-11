import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    target: str
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    cyclic_features: dict[str,int] = field(default_factory=dict)
    drop_features: list[str] = field(default_factory=list)
    target_log_transform: bool = False


DATASETS = {
    "breast_cancer": DatasetConfig(
        name="Breast Cancer",
        target="class",
        numeric_features=[
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
        ],
        drop_features=["Sample Code Number"],
    ),

    "car": DatasetConfig(
        name="Car Evaluation",
        target="class",
        numeric_features=[],
        categorical_features=[
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
        ],
    ),

    "votes": DatasetConfig(
        name="Congressional Vote",
        target="class",
        numeric_features=[],
        categorical_features=[
            "handicapped-infants",
            "water-project-cost-sharing",
            "adoption-of-the-budget-resolution",
            "physician-fee-freeze",
            "el-salvador-aid",
            "religious-groups-in-schools",
            "anti-satellite-test-ban",
            "aid-to-nicaraguan-contras",
            "mx-missile",
            "immigration",
            "synfuels-corporation-cutback",
            "education-spending",
            "superfund-right-to-sue",
            "crime",
            "duty-free-exports",
            "export-administration-act-south-africa",
        ],
    ),

    "abalone": DatasetConfig(
        name="Abalone",
        target="Rings",
        numeric_features=[
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Visceral weight",
            "Shell weight",
        ],
        # Sex is discarded because the assignment explicitly permits this.
        drop_features=["Sex"],
    ),

    "computer_hardware": DatasetConfig(
        name="Computer Hardware",
        target="PRP",
        numeric_features=[
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
        ],
        # Vendor, model, and ERP are not useful predictor features.
        drop_features=["vendor", "model", "ERP"],
    ),

    "forest_fires": DatasetConfig(
        name="Forest Fires",
        target="area",
        numeric_features=[
            "X",
            "Y",
            "FFMC",
            "DMC",
            "DC",
            "ISI",
            "temp",
            "RH",
            "wind",
            "rain",
        ],
        categorical_features=[],
        cyclic_features={
            "month": 12,
            "day": 7,
        },
        target_log_transform=True,
    )
}


def load_dataset(name, data_dir="datasets"):
    #load a dataset
    #returns
    #X: predictor
    #y: target
    #config

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: '{name}' .")

    config = DATASETS[name]

    if name == "breast_cancer":
        columns = [
            "Sample Code Number",
            "Clump Thickness",
            "Uniformity of Cell Size",
            "Uniformity of Cell Shape",
            "Marginal Adhesion",
            "Single Epithelial Cell Size",
            "Bare Nuclei",
            "Bland Chromatin",
            "Normal Nucleoli",
            "Mitoses",
            "class",
        ]

        df = pd.read_csv(f"{data_dir}/breast-cancer-wisconsin.data", names=columns)
        df = df.dropna().reset_index(drop=True)

    elif name == "car":
        columns = [
            "buying",
            "maint",
            "doors",
            "persons",
            "lug_boot",
            "safety",
            "class",
        ]

        df = pd.read_csv(
            f"{data_dir}/car.data",
            names=columns,
        )

    elif name == "votes":
        columns = [
            "class",
            "handicapped-infants",
            "water-project-cost-sharing",
            "adoption-of-the-budget-resolution",
            "physician-fee-freeze",
            "el-salvador-aid",
            "religious-groups-in-schools",
            "anti-satellite-test-ban",
            "aid-to-nicaraguan-contras",
            "mx-missile",
            "immigration",
            "synfuels-corporation-cutback",
            "education-spending",
            "superfund-right-to-sue",
            "crime",
            "duty-free-exports",
            "export-administration-act-south-africa",
        ]

        df = pd.read_csv(
            f"{data_dir}/house-votes-84.data",
            names=columns,
        )

    elif name == "abalone":
        columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Visceral weight",
            "Shell weight",
            "Rings",
        ]

        df = pd.read_csv(
            f"{data_dir}/abalone.data",
            names=columns,
        )

    elif name == "computer_hardware":
        columns = [
            "vendor",
            "model",
            "MYCT",
            "MMIN",
            "MMAX",
            "CACH",
            "CHMIN",
            "CHMAX",
            "PRP",
            "ERP",
        ]

        df = pd.read_csv(
            f"{data_dir}/machine.data",
            names=columns,
        )

    elif name == "forest_fires":
        columns = [
            "X",
            "Y",
            "month",
            "day",
            "FFMC",
            "DMC",
            "DC",
            "ISI",
            "temp",
            "RH",
            "wind",
            "rain",
            "area",
        ]

        df = pd.read_csv(
            f"{data_dir}/forestfires.csv",
            names=columns,
            skiprows=1,
        )

        # Convert month/day to numeric cycle positions.
        month_map = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }

        day_map = {
            "sun": 1,
            "mon": 2,
            "tue": 3,
            "wed": 4,
            "thu": 5,
            "fri": 6,
            "sat": 7,
        }

        df["month"] = df["month"].str.lower().map(month_map)
        df["day"] = df["day"].str.lower().map(day_map)

    else:
        raise RuntimeError("Dataset loader is missing.")


    columns_to_drop = [
        col for col in config.drop_features
        if col in df.columns
    ]

    X = df.drop(columns=[config.target] + columns_to_drop)
    y = df[config.target]

    if config.target_log_transform:
        y = np.log1p(y.astype(float))

    return X.reset_index(drop=True), y.reset_index(drop=True), config
