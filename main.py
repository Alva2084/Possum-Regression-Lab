# Adolfo Alvarez Jr

# Do the thing
from regression import loadPossumData, predictHeadLength, topCorrelations, populationEvaluationClassifier

if __name__ == "__main__":
    data = loadPossumData("possum.csv")

    # Regression
    reg = predictHeadLength(data)
    print("Regression Results:")
    print(reg.regressionEquation)
    print("R2:", reg.r2Score)
    print("RMSE:", reg.rootMeanSquaredError)
    print()

    # Correlations
    age_corr, sex_corr = topCorrelations(data)
    print("Top Age Correlations:")
    print(age_corr.head(3))
    print()
    print("Top Sex Correlations:")
    print(sex_corr.head(3))
    print()

    # Population classifier
    pop = populationEvaluationClassifier(data)
    print("Population Classification:")
    print("Accuracy:", pop.accuracy)
    print("Macro F1:", pop.macroF1Score)