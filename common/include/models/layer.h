#pragma once

#include "point.h"

#include <string>
#include <vector>

struct Layer
{
    std::vector<Point> points{};
    std::string composition{};
};
