
#pragma once
#include <eigen3/Eigen/Dense>
#include <models/point.h>
#include <models/lithology_data.h>
#include "models.h"

Eigen::MatrixXd calculateCovarianceMatrix(const std::vector<Point>* observedData, TheoreticalParam param);

KrigingOutput kriging(const std::vector<Point>* observedData, TheoreticalParam param, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY);

void createInterpolation(const std::vector<Point>* observedData, LithologyData* lithoData, WorkingArea* area);

double computeConditionNumber(const Eigen::MatrixXd& R);

Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax);
