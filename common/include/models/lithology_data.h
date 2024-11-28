#pragma once

#include "point.h"
#include <vector>
#include <string>

struct EmpiricalVariogram {
    std::vector<double> values;
    std::vector<double> distances;
};
struct TheoreticalParam {
    double nugget;
    double sill;
    double range;
};
struct Variogram
{
    EmpiricalVariogram empirical;
    TheoreticalParam theoretical;
};

struct LithologyData {
    float averageDepth;
    std::string stratumName;
    std::vector<Point> points;
    Variogram variogram;
    std::vector<Point> interpolatedData;
    std::vector<Point> certaintyMatrix;
};
struct BoundingRectangle {
    double maxX;
    double maxY;
    double minX;
    double minY;
};
struct WorkingArea {
    BoundingRectangle boundingRect;
    int xAxisPoints;
    int yAxisPoints;
    double xScale;
    double yScale;
};