
#pragma once
#include <eigen3/Eigen/Dense>
#include <models/point.h>
#include <models/lithology_data.h>
#include "models.h"

Eigen::MatrixXd calculateCovarianceMatrix(std::vector<Point>& observedData, TheoreticalParam param);

KrigingOutput kriging(LithologyData& lithoData, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY);

void createInterpolation(LithologyData &lithoData, WorkingArea &area) ;

double computeConditionNumber(const Eigen::MatrixXd& R);

Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax);
