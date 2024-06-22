#include <geotiff_handler.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

std::vector<Point> processGeoTIFF(const std::string& filename)
{
    // Register GDAL drivers
    GDALAllRegister();

    // Open the GeoTIFF file
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "Error: Unable to open GeoTIFF file." << std::endl;
        return {};
    }

    // Get raster band (assuming single band for elevation)
    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    int nXSize = poBand->GetXSize();
    int nYSize = poBand->GetYSize();

    // Read the elevation data into a buffer
    float* pafScanline = (float*)CPLMalloc(sizeof(float) * nXSize * nYSize);
    poBand->RasterIO(GF_Read, 0, 0, nXSize, nYSize, pafScanline, nXSize, nYSize, GDT_Float32, 0, 0);

    // Convert the buffer to an OpenCV Mat
    cv::Mat elevationMat(nYSize, nXSize, CV_32F, pafScanline);
    cv::Mat blurred;

    // Get GeoTransform to convert pixel coordinates to real-world coordinates
    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);

    // Create a vector to store points
    std::vector<Point> points;

    // Iterate through each pixel in the blurred image
    for (int y = 0; y < blurred.rows; ++y) {
        for (int x = 0; x < blurred.cols; ++x) {
            float elevation = blurred.at<float>(y, x);

            // Convert pixel coordinates to real-world coordinates
            double realX = adfGeoTransform[0] + x * adfGeoTransform[1] + y * adfGeoTransform[2];
            double realY = adfGeoTransform[3] + x * adfGeoTransform[4] + y * adfGeoTransform[5];

            Point p;
            p.x = realX;
            p.y = realY;
            p.z = elevation;

            points.push_back(p);
        }
    }

    // Clean up
    CPLFree(pafScanline);
    GDALClose(poDataset);

    return points;
}


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
