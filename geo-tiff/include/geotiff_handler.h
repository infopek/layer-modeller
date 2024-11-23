#pragma once

#include <models/point.h>
#include <models/lithology_data.h>
#include <logging.h>

#include <gdal_priv.h>

#include <stdexcept>
#include <string>

class GeoTiffHandler
{
public:
    GeoTiffHandler(const std::string& filepath);
    ~GeoTiffHandler();

    BoundingRectangle getBoundingRectangle();
    float* getRaster();
    void blurRaster(float* raster);
    void freeRaster(float* raster);

private:
    void handleOrientation();
    void flipVertically();
    GDALDataset* m_dataset;
    int nXSize;
    int nYSize;
    float* pafRaster;
    static std::string s_logPrefix;
};
