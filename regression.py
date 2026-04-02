# Adolfo Alvarez Jr

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

bodyDimensions = [
    "hdlngth",
    "skullw",
    "totlngth",
    "taill",
    "footlgth",
    "earconch",
    "eye",
    "chest",
    "belly",
]
"""
This file contains functions to perform regression and classification tasks on the
possum dataset.
"""

@dataclass(frozen=True)
class RegressionResults:
    regressionEquation: str
    r2Score: float
    rootMeanSquaredError: float


@dataclass(frozen=True)
class ClassificationResults:
    accuracy: float
    macroF1Score: float

# Load the possum dataset from the CSV file
def loadPossumData(csvFilePath: str) -> pd.DataFrame:
    csv_path = Path(csvFilePath)
    if not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path
    return pd.read_csv(csv_path)

# Compute the absolute correlations of body dimensions with age and sex
def topCorrelations(possumData: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    ageCorrelation = (
        possumData[bodyDimensions + ["age"]]
        .corr(numeric_only = True)["age"]
        .drop("age")
        .abs()
        .sort_values(ascending = False)
    )
    sexNumeric = possumData["sex"].map({"m": 1, "f": 0})
    sexCorrelation = (
        pd.concat([possumData[bodyDimensions], sexNumeric.rename("sex")], axis=1)
        .corr(numeric_only=True)["sex"]
        .drop("sex")
        .abs()
        .sort_values(ascending=False)
    )
    return ageCorrelation, sexCorrelation

# Predicts the head length using regression
def predictHeadLength(possumData: pd.DataFrame) -> RegressionResults:
    """Fit a simple linear regression model to predict head length from total length."""
    regressionData = possumData[["totlngth", "hdlngth"]].dropna()
    predictorValues = regressionData[["totlngth"]]
    targetValues = regressionData["hdlngth"]

    linearModel = LinearRegression()
    linearModel.fit(predictorValues, targetValues)

    predictedHeadLengths = linearModel.predict(predictorValues)
    intercept = linearModel.intercept_
    slope = linearModel.coef_[0]

    regressionEquation = f"hdlngth = {intercept:.3f} + {slope:.3f} * totlngth"
    r2ScoreValue = r2_score(targetValues, predictedHeadLengths)
    rootMeanSquaredErrorValue = sqrt(mean_squared_error(targetValues, predictedHeadLengths))

    return RegressionResults(
        regressionEquation = regressionEquation,
        r2Score = r2ScoreValue,
        rootMeanSquaredError = rootMeanSquaredErrorValue,
    )

# Builds the logic for the sex prediction classifier
def buildSexClassifier() -> Pipeline:
    categoricalColumns = ["site", "Pop"]
    numericColumns = bodyDimensions

    preprocessing = ColumnTransformer(
        transformers = [
            (
                "categorical",
                Pipeline(
                    steps = [
                        ("imputer", SimpleImputer(strategy = "most_frequent")), (
                            "onehot",
                            OneHotEncoder(handle_unknown = "ignore", sparse_output=False),
                        ),
                    ]
                ),
                categoricalColumns,
            ),
            (
                "numeric",
                Pipeline (
                    steps = [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numericColumns,
            ),
        ]
    )

    return Pipeline (
        steps = [
            ("preprocessing", preprocessing), (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=42),
            ),
        ]
    )

# Train and evaluate a logistic regression model for population prediction
def populationEvaluationClassifier(possumData: pd.DataFrame) -> ClassificationResults:
    """Train and evaluate a logistic regression model for population prediction."""
    featureColumns = bodyDimensions + ["site"]
    predictorValues = possumData[featureColumns]
    targetValues = possumData["Pop"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                ["site"],
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                bodyDimensions,
            ),
        ]
    )

    populationClassifier = Pipeline(
        steps = [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=42),
            ),
        ]
    )

    populationClassifier.fit(predictorValues, targetValues)
    predictedPopulation = populationClassifier.predict(predictorValues)

    return ClassificationResults(
        accuracy = accuracy_score(targetValues, predictedPopulation),
        macroF1Score = f1_score(targetValues, predictedPopulation, average="macro"),
    )


__all__ = [
    "ClassificationResults",
    "RegressionResults",
    "bodyDimensions",
    "buildSexClassifier",
    "loadPossumData",
    "populationEvaluationClassifier",
    "predictHeadLength",
    "topCorrelations",
]
