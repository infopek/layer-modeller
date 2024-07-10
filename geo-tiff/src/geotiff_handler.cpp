#include <geotiff_handler.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <blur/blur.h>

#include <iostream>

std::string GeoTiffHandler::s_logPrefix = "[GEO_TIFF_HANDLER] --";

GeoTiffHandler::GeoTiffHandler(const std::string& filepath)
{
    GDALAllRegister();

    m_dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (m_dataset == nullptr)
    {
        Logger::log(LogLevel::CRITICAL, GeoTiffHandler::s_logPrefix + " Failed to open GeoTIFF file.");
        throw std::runtime_error("Failed to open GeoTIFF file.");
    }
}

GeoTiffHandler::~GeoTiffHandler()
{
    if (m_dataset)
    {
        Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Closing dataset...");
        GDALClose(m_dataset);
    }
}

BoundingRectangle GeoTiffHandler::getBoundingRectangle()
{
    double adfGeoTransform[6];
    if (m_dataset->GetGeoTransform(adfGeoTransform) != CE_None)
    {
        Logger::log(LogLevel::CRITICAL, GeoTiffHandler::s_logPrefix + " Failed to get geotransform.");
        throw std::runtime_error("Failed to get geotransform.");
    }

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

float* GeoTiffHandler::getRaster()
{
    GDALRasterBand* poBand = m_dataset->GetRasterBand(1); // Get the first band
    if (poBand == nullptr)
    {
        Logger::log(LogLevel::ERROR, GeoTiffHandler::s_logPrefix + " Failed to get raster band.");
        return nullptr;
    }

    int nXSize = poBand->GetXSize();
    int nYSize = poBand->GetYSize();
    float* pafRaster = (float*)CPLMalloc(sizeof(float) * nXSize * nYSize);

    if (poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, pafRaster, nXSize, nYSize, GDT_Float32, 0, 0) != CE_None)
    {
        Logger::log(LogLevel::ERROR, GeoTiffHandler::s_logPrefix + " Failed to read raster data.");
        CPLFree(pafRaster);
        return nullptr;
    }
    return pafRaster;
}

void GeoTiffHandler::freeRaster(float* raster)
{
    Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Freeing raster...");
    CPLFree(raster);
}