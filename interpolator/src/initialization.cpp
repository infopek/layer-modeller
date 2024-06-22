#include "initialization.h"

void readObservationDataFromJson(std::map<std::string, LithologyData> &lithologyMap, std::string path, InterpolatedArea* area)
{
    std::ifstream f(path);
    json dataJson = json::parse(f);
    area->boundingRect.minX = -1;
    area->boundingRect.minY = -1;
    double minVal = -1;

    for (const auto &entry : dataJson)
    {
        Point point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<double>(entry["eovX"]);
        point.y = static_cast<double>(entry["eovY"]);
        point.z = entry["reteg$mig"].get<double>();
        lithologyMap[lithoType].points.push_back(point);
        if (lithologyMap[lithoType].stratumName.empty())
        {
            lithologyMap[lithoType].stratumName = entry["reteg$lito$nev"];
        }
        if (area->boundingRect.minX == -1 || area->boundingRect.minX > point.x)
            area->boundingRect.minX = point.x;
        ;
        if (area->boundingRect.minY == -1 || area->boundingRect.minY > point.y)
            area->boundingRect.minY = point.y;
        if (minVal == -1 || minVal > point.z)
            minVal = point.z;
        if (area->boundingRect.maxX < point.x)
            area->boundingRect.maxX = point.x;
        if (area->boundingRect.maxY < point.y)
            area->boundingRect.maxY = point.y;
    }
    double xScale = (area->boundingRect.maxX - area->boundingRect.minX) / 100;
    double yScale = (area->boundingRect.maxY - area->boundingRect.minY) / 100;
    std::cout << "maxX: " << area->boundingRect.maxX << "  maxY: " << area->boundingRect.maxY << std::endl;
    std::cout << "minX: " << area->boundingRect.minX << "   minY: " << area->boundingRect.minY << std::endl;
    std::cout << "scale: x" << xScale << ", y" << yScale << std::endl;

    int width = static_cast<int>(area->boundingRect.maxX - area->boundingRect.minX);
    int height = static_cast<int>(area->boundingRect.maxY - area->boundingRect.minY);

    int gcd = std::gcd(width, height);
    int aspectWidth = width / gcd;
    int aspectHeight = height / gcd;

    int numRows = static_cast<int>(std::sqrt(10000 / (aspectWidth / static_cast<double>(aspectHeight))));
    int numCols = 10000 / numRows;

    if (numRows * numCols > 10000)
        numCols = 10000 / numRows;

    std::cout << "Adjusted xAxisPoints: " << numRows << ", yAxisPoints: " << numCols << std::endl;
    area->xAxisPoints = numCols;
    area->yAxisPoints = numRows;
}
