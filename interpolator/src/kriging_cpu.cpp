#include "kriging_cpu.h"
#include "kriging_utilities.cuh"

Eigen::MatrixXd calculateCovarianceMatrix(const std::vector<DataPoint>* observedData, TheoreticalParam param) {
    int n = observedData->size();
    Eigen::MatrixXd covMatrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double h = std::sqrt(std::pow(observedData->at(i).x - observedData->at(j).x, 2) + std::pow(observedData->at(i).y - observedData->at(j).y, 2));
            covMatrix(i, j) = gaussianFunction(param.nugget, param.sill, param.range, h);
            printf("%4.2f\t", covMatrix(i, j));
        }
        printf("\n");
    }
    return covMatrix;
}
double kriging(const std::vector<DataPoint>* observedData, TheoreticalParam param, const Eigen::MatrixXd& covMatrix, double targetX, double targetY) {
    int n = observedData->size();
    Eigen::VectorXd k(n);
    for (int i = 0; i < n; ++i) {
        double h = std::sqrt(std::pow(observedData->at(i).x - targetX, 2) + std::pow(observedData->at(i).y - targetY, 2));
        k(i) = gaussianFunction(param.nugget, param.sill, param.range, h);
    }

    Eigen::VectorXd weights = covMatrix.fullPivLu().solve(k);
    double estimatedValue = 0.0;
    for (int i = 0; i < n; ++i) {
        estimatedValue += weights(i) * observedData->at(i).value;
    }
    return estimatedValue;
}
void createInterpolation(const std::vector<DataPoint>* observedData, TheoreticalParam param, Eigen::MatrixXd* krigingOutput, int maxX, int maxY) {
    Eigen::MatrixXd covMatrix = calculateCovarianceMatrix(observedData, param);
    for (double i = 0, nRows = krigingOutput->rows(), nCols = krigingOutput->cols(); i < nRows; ++i) {
        for (double j = 0; j < nCols; ++j) {
            (*krigingOutput)((int)j, (int)i) = kriging(observedData, param, covMatrix, (i / nRows) * maxX, (j / nCols) * maxY);
        }
    }
}
