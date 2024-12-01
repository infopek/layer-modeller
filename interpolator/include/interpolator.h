#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "plotting.h"
#include "variogram.h"
#include "initialization.h"
#include "kriging_cpu.h"
#include "models/point.h"
#include <models/lithology_data.h>
#include "models.h"

class Interpolator
{
public:
    std::vector<std::pair<std::string, LithologyData>> interpolate(WorkingArea &area, const std::string &observationDataPath);

private:
    double calculateAverage(const std::vector<double>& vec);
    CalculationRunTime calculateAverage(const std::vector<CalculationRunTime>& vec);
    void calculatePointDensity(WorkingArea &area, int maxPointCount);
    void validateAndGroup(
        const std::vector<std::pair<std::string, LithologyData>>& lithologyVector,
        const std::vector<Point>& testLayer);
};
