#pragma once
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
