#include <geotiff_handler.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <blur/blur.h>

#include <iostream>

GeoTiffHandler::GeoTiffHandler(const std::string& filepath)
{
    GDALAllRegister();

    m_dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (m_dataset == nullptr)
        throw std::runtime_error("Failed to open GeoTIFF file.");
}

GeoTiffHandler::~GeoTiffHandler()
{
    if (m_dataset)
        GDALClose(m_dataset);
}

BoundingRectangle GeoTiffHandler::getBoundingRectangle()
{
    double adfGeoTransform[6];
    if (m_dataset->GetGeoTransform(adfGeoTransform) != CE_None)
        throw std::runtime_error("Failed to get geotransform.");

    // Image size
    int xSize = m_dataset->GetRasterXSize();
    int ySize = m_dataset->GetRasterYSize();

    // Corner coordinates
    double minX = adfGeoTransform[0];
    double maxX = adfGeoTransform[0] + xSize * adfGeoTransform[1] + ySize * adfGeoTransform[2];
    double minY = adfGeoTransform[3] + xSize * adfGeoTransform[4] + ySize * adfGeoTransform[5];
    double maxY = adfGeoTransform[3];

    BoundingRectangle boundingRect;
    boundingRect.minX = minX;
    boundingRect.minY = minY;
    boundingRect.maxX = maxX;
    boundingRect.maxY = maxY;

    return boundingRect;
}
