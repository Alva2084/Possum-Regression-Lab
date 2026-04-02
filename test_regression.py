# Adolfo Alvarez Jr

from regression import (
    loadPossumData,
    topCorrelations,
    predictHeadLength,
    buildSexClassifier,
    populationEvaluationClassifier,
    bodyDimensions,
)

"""
This file tests the regression and classification functions
for the possum dataset.
"""

POSSUM_DATA_CSV = "possum.csv"

# Tests that the function properly loads the correct columns and number of
# rows from the possum dataset.
def test_expected_data_columns_to_appear():
    possumData = loadPossumData(POSSUM_DATA_CSV)
    requiredColumns = {"sex", "age", "site", "Pop", "hdlngth", "totlngth"}
    assert requiredColumns.issubset(possumData.columns)
    assert len(possumData) == 104

# Tests that the regression equation predicts the head length with accuracy
def test_head_length_prediction():
    possumData = loadPossumData(POSSUM_DATA_CSV)
    regressionResults = predictHeadLength(possumData)
    assert "totlngth" in regressionResults.regressionEquation
    assert regressionResults.r2Score > 0.30
    assert regressionResults.rootMeanSquaredError < 3.0

# Tests the correlation ranking of the body dimensions with age and sex
def test_dimension_correlation():
    possumData = loadPossumData(POSSUM_DATA_CSV)
    ageCorrelation, sexCorrelation = topCorrelations(possumData)
    assert list(ageCorrelation.index[:3]) == ["belly", "chest", "hdlngth"]
    assert sexCorrelation.index[0] == "eye"
    assert set(ageCorrelation.index) == set(bodyDimensions)
    assert set(sexCorrelation.index) == set(bodyDimensions)

# Tests that the sex classification can make a valid prediction
def test_predict_sex():
    possumData = loadPossumData(POSSUM_DATA_CSV)
    model = buildSexClassifier()
    model.fit(possumData[bodyDimensions + ["site", "Pop"]], possumData["sex"])
    predictSex = model.predict(possumData[bodyDimensions + ["site", "Pop"]].head(5))
    assert len(predictSex) == 5
    assert set(predictSex).issubset({"m", "f"})

# Tests that the population evaluation classifier achieves high accuracy and F1 score
def test_population_classifier():
    possumData = loadPossumData(POSSUM_DATA_CSV)
    populationResults = populationEvaluationClassifier(possumData)
    assert populationResults.accuracy > 0.90
    assert populationResults.macroF1Score > 0.90
