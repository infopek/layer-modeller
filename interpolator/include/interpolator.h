#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include "plotting.h"
#include "variogram.h"
#include "initialization.h"
#include "kriging_cpu.h"
#include "models/point.h"
#include <models/lithologyData.h>
#include "models.h"
std::map<std::string, LithologyData> interpolate();