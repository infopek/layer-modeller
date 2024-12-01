#include <geotiff_handler.h>
#include <blur/blur.h>
#include <iostream>
#include <algorithm> // For std::swap

std::string GeoTiffHandler::s_logPrefix = "[GEO_TIFF_HANDLER] --";

GeoTiffHandler::GeoTiffHandler(const std::string &filepath)
{
    GDALAllRegister();

    m_dataset = (GDALDataset *)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (m_dataset == nullptr)
    {
        Logger::log(LogLevel::CRITICAL, GeoTiffHandler::s_logPrefix + " Failed to open GeoTIFF file.");
        throw std::runtime_error("Failed to open GeoTIFF file.");
    }
    GDALRasterBand *poBand = m_dataset->GetRasterBand(1);
    if (poBand == nullptr)
    {
        Logger::log(LogLevel::ERROR, GeoTiffHandler::s_logPrefix + " Failed to get raster band.");
        pafRaster = nullptr;
    }

    nXSize = poBand->GetXSize();
    nYSize = poBand->GetYSize();
    pafRaster = (float *)CPLMalloc(sizeof(float) * nXSize * nYSize);

    if (poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, pafRaster, nXSize, nYSize, GDT_Float32, 0, 0) != CE_None)
    {
        Logger::log(LogLevel::ERROR, GeoTiffHandler::s_logPrefix + " Failed to read raster data.");
        CPLFree(pafRaster);
        pafRaster = nullptr;
    }

    handleOrientation();
}

void GeoTiffHandler::handleOrientation()
{
    double geoTransform[6];
    if (m_dataset->GetGeoTransform(geoTransform) == CE_None)
    {
        Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Checking raster orientation...");

        if (geoTransform[5] < 0)
        {
            Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Raster is flipped vertically. Correcting...");
            flipVertically();
        }
    }
    else
    {
        Logger::log(LogLevel::WARN, GeoTiffHandler::s_logPrefix + " Unable to determine geotransform for orientation.");
    }
}

void GeoTiffHandler::flipVertically()
{
    if (pafRaster == nullptr) return;
    int rowSize = nXSize; 
    float *tempRow = (float *)CPLMalloc(rowSize * sizeof(float)); 

    for (int row = 0; row < nYSize / 2; ++row)
    {
        int oppositeRow = nYSize - row - 1;
        std::memcpy(tempRow, pafRaster + row * rowSize, rowSize * sizeof(float));
        std::memcpy(pafRaster + row * rowSize, pafRaster + oppositeRow * rowSize, rowSize * sizeof(float));
        std::memcpy(pafRaster + oppositeRow * rowSize, tempRow, rowSize * sizeof(float));
    }
    CPLFree(tempRow);
    Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Raster flip complete.");
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

float *GeoTiffHandler::getRaster()
{
    return pafRaster;
}

void GeoTiffHandler::blurRaster(float *raster)
{
    Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Applying blur to TIFF.");
    Blur::medianFilter(pafRaster, pafRaster, nXSize, nYSize, 7);
    Blur::gaussFilter(pafRaster, pafRaster, nXSize, nYSize, 7, 2.4f);
}

void GeoTiffHandler::freeRaster(float *raster)
{
    Logger::log(LogLevel::INFO, GeoTiffHandler::s_logPrefix + " Freeing raster...");
    CPLFree(raster);
}