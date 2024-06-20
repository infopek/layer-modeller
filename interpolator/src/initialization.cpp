#include "initialization.h"

void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap, std::string path, BoundingRectangle* boundingRect) {
    std::ifstream f(path);
    json dataJson = json::parse(f);
    boundingRect->minX = -1;
    boundingRect->minY = -1;
    double minVal = -1;

    for (const auto& entry : dataJson) {
        Point point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<double>(entry["eovX"]);
        point.y = static_cast<double>(entry["eovY"]);
        point.z = entry["reteg$mig"].get<double>();
        lithologyMap[lithoType].points.push_back(point);
        if (lithologyMap[lithoType].stratumName.empty()) {
            lithologyMap[lithoType].stratumName = entry["reteg$lito$nev"];
        }
        if (boundingRect->minX == -1 || boundingRect->minX > point.x)
            boundingRect->minX = point.x;
            ;
        if (boundingRect->minY == -1 || boundingRect->minY > point.y)
            boundingRect->minY = point.y;
        if (minVal == -1 || minVal > point.z)
            minVal = point.z;
        if ( boundingRect->maxX < point.x)
             boundingRect->maxX = point.x;
        if ( boundingRect->maxY < point.y)
             boundingRect->maxY = point.y;
    }

    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        int numRows = 100; // Example size, adjust based on actual needs
        int numCols = 100; // Example size, adjust based on actual needs

        // Further processing like interpolation and certainty computation would go here
    }
            double xScale=(boundingRect->maxX-boundingRect->minX)/100;
        double yScale=(boundingRect->maxY-boundingRect->minY)/100;
        std::cout<<"maxX: "<<boundingRect->maxX<<"  maxY: "<< boundingRect->maxY<<std::endl;
        std::cout<<"minX: "<<boundingRect->minX<<"   minY: "<< boundingRect->minY<<std::endl;
        std::cout<<"scale: x"<<xScale<<", y"<<yScale<<std::endl;
}
