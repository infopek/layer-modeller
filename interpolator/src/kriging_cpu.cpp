#include "kriging_cpu.h"
#include <iostream>

Eigen::MatrixXd calculateCovarianceMatrix(const std::vector<Point>* observedData, TheoreticalParam param) {
    int n = observedData->size();
    Eigen::MatrixXd covMatrix(n, n);
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double h = std::sqrt(std::pow(observedData->at(i).x - observedData->at(j).x, 2) + std::pow(observedData->at(i).y - observedData->at(j).y, 2));
            covMatrix(i, j) = nugget + sill * (1.0 - exp(-((h * h) / (range * range))));
            printf("%4.2f\t", covMatrix(i, j));
        }
        printf("\n");
    }
    return covMatrix;
}
KrigingOutput kriging(const std::vector<Point>* observedData, TheoreticalParam param, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY) {
    int n = observedData->size();
    Eigen::VectorXd k(n);
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (int i = 0; i < n; ++i) {
        double h = std::sqrt(std::pow(observedData->at(i).x - targetX, 2) + std::pow(observedData->at(i).y - targetY, 2));
        k(i) = nugget + sill * (1.0 - exp(-((h * h) / (range * range))));
    }
    Eigen::VectorXd weights = luCovMatrix.solve(k);
    double estimatedValue = 0.0;
    for (int i = 0; i < n; ++i) {
        estimatedValue += weights(i) * observedData->at(i).z;
    }
    KrigingOutput output;
    output.value = estimatedValue;
    //output.certainty = sqrt((weights.transpose() * k)[0]);
    return output;
}
void createInterpolation(const std::vector<Point>* observedData, LithologyData* lithoData, WorkingArea* area) {
    Eigen::MatrixXd covMatrix = calculateCovarianceMatrix(observedData, lithoData->theoreticalParam);
    int n = observedData->size();
    const BoundingRectangle bRect=area->boundingRect;
    double condR = computeConditionNumber(covMatrix);
    std::cout << "Condition number of cov mx: " << condR << std::endl;

    double kmax = pow(condR, 2) * 100;
    std::cout << "Kmax: " << kmax << std::endl;
    Eigen::MatrixXd regularizedCovMatrix =  ridgeRegression(covMatrix, kmax );
    Eigen::FullPivLU<Eigen::MatrixXd> luCovMatrix = regularizedCovMatrix.fullPivLu();

    for (double i = 0; i < area->yAxisPoints; ++i) {
        for (double j = 0; j < area->xAxisPoints; ++j) {
            double realY=bRect.minY+i*area->yScale;
            double realX=bRect.minX+j*area->xScale;
            KrigingOutput output= kriging(observedData, lithoData->theoreticalParam, luCovMatrix, realX,realY);
            Point pointValue(realX,realY,-output.value);
            Point pointCertainty(realX,realY,output.certainty);
            lithoData->interpolatedData.push_back(pointValue);
        }
    }
}
double computeConditionNumber(const Eigen::MatrixXd& R) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    double lambda1 = eigenvalues(eigenvalues.size() - 1);
    double lambdad = eigenvalues(0);
    return (lambda1 / lambdad);
    //return cond;
}
Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    double lambda1 = eigenvalues(eigenvalues.size() - 1);
    double lambdad = eigenvalues(0); 

    double delta = (lambda1 - lambdad) / (kmax - 1);

    Eigen::MatrixXd R_RR = R + delta * Eigen::MatrixXd::Identity(R.rows(), R.cols());

    return R_RR;
}