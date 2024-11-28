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
    void calculatePointDensity(WorkingArea &area, int maxPointCount);
};
