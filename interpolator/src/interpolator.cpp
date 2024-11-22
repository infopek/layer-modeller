#include "interpolator.h"

#include <string>
#include <initialization.h>
struct InterpolationError {
    double observed;
    double predicted;
    double error;
};
std::vector<std::pair<std::string, LithologyData>> interpolate(WorkingArea& area, const std::string& observationDataPath) {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::vector<std::pair<std::string, LithologyData>> lithologyVector;
    calculatePointDensity(area, 10000);
    readObservationDataFromJson(lithologyVector, observationDataPath);
    for (auto it = lithologyVector.begin(); it != lithologyVector.end(); ++it) {
        auto& data = it->second;
        std::cout<<std::endl <<it->first<< std::endl;
        createInterpolation(data, area,true);
    }
    return lithologyVector;
}
void calculatePointDensity(WorkingArea& area, int maxPointCount) {

    int width = static_cast<int>(area.boundingRect.maxX - area.boundingRect.minX);
    int height = static_cast<int>(area.boundingRect.maxY - area.boundingRect.minY);

    int gcd = std::gcd(width, height);
    int aspectWidth = width / gcd;
    int aspectHeight = height / gcd;

    int numRows = static_cast<int>(std::sqrt(maxPointCount / (aspectWidth / static_cast<double>(aspectHeight))));
    int numCols = maxPointCount / numRows;

    if (numRows * numCols > maxPointCount)
        numCols = maxPointCount / numRows;

    area.xScale = (area.boundingRect.maxX - area.boundingRect.minX) / numCols;
    area.yScale = (area.boundingRect.maxY - area.boundingRect.minY) / numRows;
    area.xAxisPoints = numCols;
    area.yAxisPoints = numRows;
    ;
}
