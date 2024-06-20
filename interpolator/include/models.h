#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>
struct EmpiricalVariogram {
    std::vector<double> values;
    std::vector<double> distances;
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

struct BoundingRectangle{
    double maxX;
    double maxY;
    double minX;
    double minY;
};