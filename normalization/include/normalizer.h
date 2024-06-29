#pragma once

#include <map>
#include <string>
#include <vector>

#include <models/lithology_data.h>
#include <geotiff_handler.h>

void resolveCrossingLayers(std::map<std::string, LithologyData>& lithologyMap);
void shiftPointsBasedOnBlur(std::map<std::string, LithologyData>& lithologyMap, GeoTiffHandler* geoTiff, WorkingArea* area);
void normalizeLayers(std::map<std::string, LithologyData>& lithologyMap, GeoTiffHandler* geoTiff, WorkingArea* area);