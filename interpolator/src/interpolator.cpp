#include "interpolator.h"

#include <string>
#include <initialization.h>

std::vector<std::pair<std::string, LithologyData>> Interpolator::interpolate(WorkingArea& area, const std::string& observationDataPath) {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::vector<std::pair<std::string, LithologyData>> lithologyVector;
    BoundingRectangle bTest;
    bTest.maxX=100;
    bTest.maxY=100;
    bTest.minX=0;
    bTest.minY=0;
    area.boundingRect=bTest;
    calculatePointDensity(area, 10000);
    generateTestData(lithologyVector);
    //readObservationDataFromJson(lithologyVector, observationDataPath);
    KrigingCalculator kriging;
    for (auto it = lithologyVector.begin(); it != lithologyVector.end(); ++it) {
        auto& data = it->second;
        std::cout<<std::endl <<it->first<< std::endl;
        kriging.createInterpolation(data, area,true);
    }
    return lithologyVector;
}
void Interpolator::calculatePointDensity(WorkingArea& area, int maxPointCount) {

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
