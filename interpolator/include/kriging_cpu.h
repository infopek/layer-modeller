
#include <eigen3/Eigen/Dense>
#include "models.h"

Eigen::MatrixXd calculateCovarianceMatrix(const std::vector<DataPoint>* observedData, TheoreticalParam param);

KrigingOutput kriging(const std::vector<DataPoint>* observedData, TheoreticalParam param, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY);

void createInterpolation(const std::vector<DataPoint>* observedData, TheoreticalParam param, Eigen::MatrixXd* krigingOutput, Eigen::MatrixXd* krigingCertainty, int maxX, int maxY);

double computeConditionNumber(const Eigen::MatrixXd& R) ;

Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax);
