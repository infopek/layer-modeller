//#include "initialization.h"
//void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap, std::string path, int* maxX, int* maxY, double* scaleFactor) {
//    std::ifstream f(path);
//    json dataJson = json::parse(f);
//    int minX, minY;
//    readTiff(minX, minY, *maxX, *maxY);
//    int rasterWidth = abs( *maxX - minX);
//    int rasterHeight = abs(*maxY - minY);
//
//    std::cout << rasterHeight << " " << rasterWidth << std::endl;
//    (*scaleFactor) = std::min(128.0 / rasterWidth, 128.0 / rasterHeight);
//    int numRows = std::min(128, static_cast<int>(*scaleFactor * rasterHeight));
//    int numCols = std::min(128, static_cast<int>(*scaleFactor * rasterWidth));
//    double minVal = -1;
//
//    for (const auto& entry : dataJson) {
//        DataPoint point;
//        std::string lithoType = entry["reteg$lito$geo"];
//        point.x = static_cast<int>(entry["eovX"]);
//        point.y = static_cast<int>(entry["eovY"]);
//        point.value = entry["reteg$mig"].get<double>();
//        lithologyMap[lithoType].points.push_back(point);
//        if (lithologyMap[lithoType].stratumName.empty()) {
//            lithologyMap[lithoType].stratumName = entry["reteg$lito$nev"];
//        }
//        if (minX == -1 || minX > point.x)
//            minX = point.x;
//        if (minY == -1 || minY > point.y)
//            minY = point.y;
//        if (minVal == -1 || minVal > point.value)
//            minVal = point.value;
//        if ((*maxX) < point.x)
//            (*maxX) = point.x;
//        if ((*maxY) < point.y)
//            (*maxY) = point.y;
//    }
//    (*maxX) = (*maxX) - minX;
//    (*maxY) = (*maxY) - minY;
//
//    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
//        auto& lithoType = it->first;
//        auto& data = it->second;
//        std::cout << numRows << " " << numCols << std::endl;
//        for (auto& point : data.points) {
//            std::cout << "point " << point.x << " " << point.y << " " << minY << std::endl;
//            point.x -= minX;
//            point.y -= minY;
//            std::cout << "point " << point.x << " " << point.y<<" "<<minY << std::endl;
//        }
//        data.interpolatedData = Eigen::MatrixXd::Zero(numRows, numCols);
//        data.certaintyMatrix = Eigen::MatrixXd::Zero(numRows, numCols);
//
//        // Further processing like interpolation and certainty computation would go here
//    }
//}
//void readTiff(int& minX, int& minY, int& maxX, int& maxY) {
//    GDALDataset* poDataset;
//
//    GDALAllRegister(); // Register all GDAL drivers
//
//    // Open the dataset
//    poDataset = (GDALDataset*)GDALOpen("Pecs_DTM_1m_EOV.tif", GA_ReadOnly);
//    if (poDataset == NULL) {
//        std::cerr << "GDAL Open failed" << std::endl;
//        return;
//    }
//
//    // Get raster dimensions
//    int nXSize = poDataset->GetRasterXSize();
//    int nYSize = poDataset->GetRasterYSize();
//
//    // Get geotransformation
//    double adfGeoTransform[6];
//    if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
//        minX = adfGeoTransform[0];
//        minY = adfGeoTransform[3];
//        maxX = adfGeoTransform[0] + nXSize * adfGeoTransform[1] + nYSize * adfGeoTransform[2];
//        maxY = adfGeoTransform[3] + nXSize * adfGeoTransform[4] + nYSize * adfGeoTransform[5];
//    }
//
//    // Close the dataset
//    GDALClose(poDataset);
//}
#include "initialization.h"
void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap, std::string path, int* maxX, int* maxY) {
    std::ifstream f(path);
    json dataJson = json::parse(f);
    int minX = -1, minY = -1;
    double minVal = -1;

    for (const auto& entry : dataJson) {
        DataPoint point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<int>(entry["eovX"]);
        point.y = static_cast<int>(entry["eovY"]);
        point.value = entry["reteg$mig"].get<double>();
        lithologyMap[lithoType].points.push_back(point);
        if (lithologyMap[lithoType].stratumName.empty()) {
            lithologyMap[lithoType].stratumName = entry["reteg$lito$nev"];
        }
        if (minX == -1 || minX > point.x)
            minX = point.x;
        if (minY == -1 || minY > point.y)
            minY = point.y;
        if (minVal == -1 || minVal > point.value)
            minVal = point.value;
        if ((*maxX) < point.x)
            (*maxX) = point.x;
        if ((*maxY) < point.y)
            (*maxY) = point.y;
    }
    (*maxX) = (*maxX) - minX;
    (*maxY) = (*maxY) - minY;

    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        int numRows = 100; // Example size, adjust based on actual needs
        int numCols = 100; // Example size, adjust based on actual needs

        for (auto& point : data.points) {
            point.x -= minX;
            point.y -= minY;
            //point.value -= minVal; // Uncomment if needed to normalize values
        }
        data.interpolatedData = Eigen::MatrixXd::Zero(numRows, numCols);
        data.certaintyMatrix = Eigen::MatrixXd::Zero(numRows, numCols);

        // Further processing like interpolation and certainty computation would go here
    }
}
void readTiff() {
    GDALDataset* poDataset;

    GDALAllRegister(); // Register all GDAL drivers

    // Open the dataset
    poDataset = (GDALDataset*)GDALOpen("Pecs_DTM_1m_EOV.tif", GA_ReadOnly);
    if (poDataset == NULL) {
        std::cerr << "GDAL Open failed" << std::endl;
    }

    // Get raster dimensions
    int nXSize = poDataset->GetRasterXSize();
    int nYSize = poDataset->GetRasterYSize();

    std::cout << "Dimensions: " << nXSize << " x " << nYSize << std::endl;

    // Get geotransformation
    double adfGeoTransform[6];
    if (poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
        std::cout << "Origin = (" << adfGeoTransform[0] << ", " << adfGeoTransform[3] << ")" << std::endl;
        std::cout << "End = (" << adfGeoTransform[2] << ", " << adfGeoTransform[4] << ")" << std::endl;
        std::cout << "Pixel Size = (" << adfGeoTransform[1] << ", " << adfGeoTransform[5] << ")" << std::endl;
    }
    for (auto geo : adfGeoTransform) {
        std::cout << geo << std::endl;
    }
    // Getting the metadata
    char** papszMetadata;
    papszMetadata = poDataset->GetMetadata(NULL);
    if (papszMetadata != NULL) {
        for (int i = 0; papszMetadata[i] != NULL; ++i) {
            std::cout << papszMetadata[i] << std::endl;
        }
    }

    // Close the dataset
    GDALClose(poDataset);

}
