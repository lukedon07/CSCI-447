import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    name: str
    target: str
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    cyclic_features: dict[str, int] = field(default_factory=dict)
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
        cyclic_features={
            "month": 12,
            "day": 7,
        },
        target_log_transform=True,
    ),
}


def load_dataset(name, data_dir="datasets"):
    """
    Load one of the six assignment datasets.

    Returns
    -------
    X : pandas.DataFrame
        Predictor features.
    y : pandas.Series
        Target.
    config : DatasetConfig
        Dataset-specific feature and transformation configuration.
    """

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'.")

    config = DATASETS[name]

    # ------------------------------------------------------------------
    # Breast Cancer
    # ------------------------------------------------------------------
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

        df = pd.read_csv(
            f"{data_dir}/breast-cancer-wisconsin.data",
            names=columns,
            na_values=["?"],
        )

        # The assignment permits dropping examples with missing features.
        df = df.dropna().reset_index(drop=True)

        # Bare Nuclei is a numeric/ordinal feature.
        for col in config.numeric_features:
            df[col] = pd.to_numeric(df[col], errors="raise")

    # ------------------------------------------------------------------
    # Car Evaluation
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Congressional Voting
    # ------------------------------------------------------------------
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

        # IMPORTANT:
        # "?" means abstain and is therefore a legitimate categorical value.
        df = pd.read_csv(
            f"{data_dir}/house-votes-84.data",
            names=columns,
        )

    # ------------------------------------------------------------------
    # Abalone
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Computer Hardware
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Forest Fires
    # ------------------------------------------------------------------
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

        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
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

        df["month"] = (
            df["month"]
            .astype(str)
            .str.lower()
            .map(month_map)
        )

        df["day"] = (
            df["day"]
            .astype(str)
            .str.lower()
            .map(day_map)
        )

        if df["month"].isna().any() or df["day"].isna().any():
            raise ValueError("Invalid month/day value found in Forest Fires data.")

    else:
        raise RuntimeError("Dataset loader is missing.")

    # ------------------------------------------------------------------
    # Validate configured columns.
    # ------------------------------------------------------------------
    feature_columns = (
        config.numeric_features
        + config.categorical_features
        + list(config.cyclic_features.keys())
    )

    missing_features = [
        col for col in feature_columns
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Dataset '{name}' is missing configured features: "
            f"{missing_features}"
        )

    # Convert configured numeric features explicitly.
    for col in config.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in config.cyclic_features:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Drop explicitly unwanted columns.
    columns_to_drop = [
        col for col in config.drop_features
        if col in df.columns
    ]

    X = df.drop(
        columns=[config.target] + columns_to_drop
    ).copy()

    y = df[config.target].copy()

    # Forest Fires follows the assignment's recommended log transform.
    if config.target_log_transform:
        y = np.log1p(y.astype(float))

    return (
        X.reset_index(drop=True),
        y.reset_index(drop=True),
        config,
    )
