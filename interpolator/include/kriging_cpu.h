
#include <eigen3/Eigen/Dense>
#include "models.h"

Eigen::MatrixXd calculateCovarianceMatrix(const std::vector<DataPoint>* observedData, TheoreticalParam param);

double kriging(const std::vector<DataPoint>* observedData, TheoreticalParam param, const Eigen::MatrixXd& covMatrix, double targetX, double targetY);

void createInterpolation(const std::vector<DataPoint>* observedData, TheoreticalParam param, Eigen::MatrixXd* krigingOutput, int maxX, int maxY);