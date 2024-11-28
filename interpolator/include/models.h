#pragma once
#include <vector>
#include <eigen3/Eigen/Dense>


struct KrigingOutput {
    double value;
    double certainty;
};
struct data {
    size_t n;
    double* h;
    double* y;
};