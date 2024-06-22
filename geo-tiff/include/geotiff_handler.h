#include <Point.h>

#include <gdal_priv.h>
#include <models/lithologyData.h>
#include <stdexcept>

BoundingRectangle getBoundingRectangle(std::string fileName);

std::vector<Point> processGeoTIFF(const std::string& filename);