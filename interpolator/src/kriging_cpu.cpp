#include "kriging_cpu.h"

KrigingCalculator::KrigingCalculator()
{
}

KrigingCalculator::~KrigingCalculator()
{
}

Eigen::MatrixXd KrigingCalculator::calculateCovarianceMatrix(std::vector<Point>& observedData, TheoreticalParam param) {
    int n = observedData.size();
    Eigen::MatrixXd covMatrix(n, n);
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double h = std::sqrt(std::pow(observedData.at(i).x - observedData.at(j).x, 2) + std::pow(observedData.at(i).y - observedData.at(j).y, 2));
            covMatrix(i, j) =  (sill - nugget) * exp(-h / range);
      }
    }
    return covMatrix;
}
KrigingOutput KrigingCalculator::krigingForPoint(LithologyData& lithoData, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY) {
    int n = lithoData.points.size();
    Eigen::VectorXd k(n);
    auto param = lithoData.variogram.theoretical;
    auto nugget = param.nugget;
    auto sill = param.sill;
    auto range = param.range;
    for (int i = 0; i < n; ++i) {
        double h = std::sqrt(std::pow(lithoData.points.at(i).x - targetX, 2) + std::pow(lithoData.points.at(i).y - targetY, 2));
        k(i) = (sill - nugget) * exp(-h / range);
    }
    Eigen::VectorXd weights = luCovMatrix.solve(k);
    double estimatedValue = 0.0;
    for (int i = 0; i < n; ++i) {
        estimatedValue += weights(i) * lithoData.points.at(i).z;
    }
    KrigingOutput output;
    output.value = estimatedValue;
    double variance = sill - weights.transpose() * k;

    output.certainty = sqrt(std::abs(variance));
    return output;
}
void KrigingCalculator::createVariogram(LithologyData& lithoData){
    EmpiricalVariogram empiricalData;
    VariogramCalculator variogramCalc(lithoData.points);
    lithoData.variogram=variogramCalc.getVariogram();
    gnuPlotVariogram(lithoData);
}
Eigen::FullPivLU<Eigen::MatrixXd> KrigingCalculator::createCovMatrix(LithologyData& lithoData, bool useRegularization){
    Eigen::MatrixXd covMatrix = calculateCovarianceMatrix(lithoData.points, lithoData.variogram.theoretical);
    double condR = computeConditionNumber(covMatrix);
    double kmax = pow(condR, 2) * 1000;
    Eigen::MatrixXd regularizedCovMatrix = covMatrix;
    if(useRegularization){
        regularizedCovMatrix=ridgeRegression(covMatrix, kmax);
    }
    return regularizedCovMatrix.fullPivLu();
}
void KrigingCalculator::crossValidateInterpolation(LithologyData& lithoData, WorkingArea& area, bool useRegularization) {
    double totalErrorAbs = 0.0;
    double totalErrorSquared = 0.0;
    int n = lithoData.points.size();
    std::cout<<lithoData.stratumName<<std::endl;
    std::vector<Point> data;
    for (int i = 0; i < n; ++i) {
        LithologyData tmp;
        tmp.points = lithoData.points;
        tmp.points.erase(tmp.points.begin() + i);

        createVariogram(tmp);
        Eigen::FullPivLU<Eigen::MatrixXd> luCovMatrix = createCovMatrix(tmp, useRegularization);
        double realX = lithoData.points[i].x;
        double realY = lithoData.points[i].y;
        double realZ = lithoData.points[i].z;

        KrigingOutput output = krigingForPoint(tmp, luCovMatrix, realX, realY);

        double predictionError = output.value - realZ;
        std::cout<<i<<". "<<predictionError<<std::endl;
        totalErrorAbs += std::abs(predictionError);
        totalErrorSquared += predictionError * predictionError;
        Point validation{ .x = realX, .y = realY, .z = std::abs(predictionError)};
        data.push_back(validation);
    }
    double MAE = totalErrorAbs / n;
    double RMSE = std::sqrt(totalErrorSquared / n);
    gnuPlotValidity(lithoData,area,data,useRegularization?"_regularized":"");
    // Output results
    std::cout << "Cross-Validation Results:" << std::endl;
    std::cout << "Mean Absolute Error (MAE): " << MAE << std::endl;
    std::cout << "Root Mean Squared Error (RMSE): " << RMSE << std::endl;
}

void KrigingCalculator::createInterpolation(LithologyData &lithoData, WorkingArea &area, bool useRegularization)
{
    const BoundingRectangle bRect = area.boundingRect;
    EmpiricalVariogram empiricalData;
    createVariogram(lithoData);

    Eigen::FullPivLU<Eigen::MatrixXd> luCovMatrix = createCovMatrix(lithoData, useRegularization);

    for (double i = 0; i < area.yAxisPoints; ++i) {
        for (double j = 0; j < area.xAxisPoints; ++j) {
            double realY = bRect.minY + i * area.yScale;
            double realX = bRect.minX + j * area.xScale;
            KrigingOutput output = krigingForPoint(lithoData, luCovMatrix, realX, realY);
            Point pointValue{ .x = realX, .y = realY, .z = output.value };
            Point pointCertainty{ .x = realX, .y = realY, .z = output.certainty };
            lithoData.interpolatedData.push_back(pointValue);
            lithoData.certaintyMatrix.push_back(pointCertainty);
        }
    }
    gnuPlotKriging(lithoData,area,useRegularization?"_regularized":"");
    //crossValidateInterpolation(lithoData, area,useRegularization);
}
double KrigingCalculator::computeConditionNumber(const Eigen::MatrixXd& R) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    double lambda1 = eigenvalues(eigenvalues.size() - 1);
    double lambdad = eigenvalues(0);
    return lambda1 / lambdad;
}
Eigen::MatrixXd KrigingCalculator::ridgeRegression(const Eigen::MatrixXd& R, double kmax) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(R);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    double lambda1 = eigenvalues(eigenvalues.size() - 1);
    double lambdad = eigenvalues(0);
    double delta = (lambda1 - lambdad) / (kmax - 1);
    // double delta = (lambda1 / kmax) - lambdad;
    if (delta < 0) delta = 0;

    Eigen::MatrixXd R_RR = R + delta * Eigen::MatrixXd::Identity(R.rows(), R.cols());

    return R_RR;
}