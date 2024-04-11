#pragma once
#include <vector>

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
struct data {
    size_t n;
    double* h;
    double* y;
};