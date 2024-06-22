#pragma once
#include <vector>
#include "point.h"
struct TheoreticalParam {
    double nugget;
    double sill;
    double range;
};
struct LithologyData {
    std::string stratumName;
    std::vector<Point> points;
    TheoreticalParam theoreticalParam;
    std::vector<Point> interpolatedData;
    std::vector<Point> certaintyMatrix;
};
struct BoundingRectangle{
    double maxX;
    double maxY;
    double minX;
    double minY;
};
struct WorkingArea{
    BoundingRectangle boundingRect;
    int xAxisPoints;
    int yAxisPoints;
    double xScale;
    double yScale;
};