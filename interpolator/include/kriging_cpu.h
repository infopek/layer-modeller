
#pragma once
#include <eigen3/Eigen/Dense>
#include <models/point.h>
#include <models/lithology_data.h>
#include "models.h"

Eigen::MatrixXd calculateCovarianceMatrix(std::vector<Point>& observedData, TheoreticalParam param);

KrigingOutput kriging(LithologyData& lithoData, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY);

void createInterpolation(LithologyData &lithoData, WorkingArea &area, bool useRegularization=true);

double computeConditionNumber(const Eigen::MatrixXd& R);

Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax);

Eigen::FullPivLU<Eigen::MatrixXd> createCovMatrix(LithologyData& lithoData, bool useRegularization);

void crossValidateInterpolation(LithologyData& lithoData, WorkingArea& area, bool useRegularization);