#pragma once

#include <map>
#include <string>
#include <vector>

#include <lithologyData.h>

void resolveCrossingLayers(std::map<std::string, LithologyData>& lithologyMap);
