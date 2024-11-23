#pragma once

#include <map>
#include <string>
#include <vector>

#include <models/lithology_data.h>
#include <geotiff_handler.h>
#include <plotting.h>

std::vector<Point> getFactorOfPreviosLayer(float strength, std::vector<Point> certainty);
void resolveCrossingLayers(std::vector<std::pair<std::string, LithologyData>>& lithologyVector);
void shiftPointsBasedOnBlur(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler &geoTiff,WorkingArea &area);
void normalizeLayers(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler &geoTiff,WorkingArea &area);

