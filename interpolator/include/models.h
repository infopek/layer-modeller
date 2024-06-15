#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>
struct DataPoint {
    int x;
    int y;
    double value;
};
struct EmpiricalVariogram {
    std::vector<double> values;
    std::vector<double> distances;
};
struct TheoreticalParam {
    double nugget;
    double sill;
    double range;
};
struct KrigingOutput {
    double value;
    double certainty;
};
struct data {
    size_t n;
    double* h;
    double* y;
};
struct LithologyData {
    std::string stratumName;
    std::vector<DataPoint> points;
    TheoreticalParam theoreticalParam;
    Eigen::MatrixXd interpolatedData;
    Eigen::MatrixXd certaintyMatrix;
};