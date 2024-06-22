#include "geotiff_handler.h"

BoundingRectangle getBoundingRectangle(const char* fileName) {
    GDALAllRegister();
    
    GDALDataset* dataset = (GDALDataset*)GDALOpen(fileName, GA_ReadOnly);
    if (dataset == nullptr) {
        throw std::runtime_error("Failed to open GeoTIFF file.");
    }

    double adfGeoTransform[6];
    if (dataset->GetGeoTransform(adfGeoTransform) != CE_None) {
        GDALClose(dataset);
        throw std::runtime_error("Failed to get geotransform.");
    }

    // Image size
    int xSize = dataset->GetRasterXSize();
    int ySize = dataset->GetRasterYSize();

    // Corner coordinates
    double minX = adfGeoTransform[0];
    double maxX = adfGeoTransform[0] + xSize * adfGeoTransform[1] + ySize * adfGeoTransform[2];
    double minY = adfGeoTransform[3] + xSize * adfGeoTransform[4] + ySize * adfGeoTransform[5];
    double maxY = adfGeoTransform[3];

    GDALClose(dataset);

    BoundingRectangle boundingRect;
    boundingRect.minX = minX;
    boundingRect.minY = minY;
    boundingRect.maxX = maxX;
    boundingRect.maxY = maxY;

    return boundingRect;
}
