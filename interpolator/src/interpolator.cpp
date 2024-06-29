#include "interpolator.h"

#include <string>

std::map<std::string, LithologyData> interpolate(WorkingArea* area, const std::string& observationDataPath) {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::map<std::string, LithologyData> lithologyMap;
    calculatePointDensity(area, 10000);
    readObservationDataFromJson(lithologyMap, observationDataPath);
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        EmpiricalVariogram empiricalData;

        std::cout << lithoType << std::endl << std::endl;
        createVariogram(&data.points, &empiricalData);

        data.theoreticalParam = fitTheoreticalFunction(&empiricalData);

        //createInterpolationGPU(&data, param, &krigingData, maxX, maxY, plotSize);

        createInterpolation(&data.points, &data, area);

        // gnuPlotMatrix("values", data.interpolatedData, data.stratumName, &data.points, maxX, maxY);
        // gnuPlotMatrix("certainty", data.certaintyMatrix, data.stratumName, &data.points, maxX, maxY);
        gnuPlotVariogram(data.stratumName, &empiricalData, data.theoreticalParam);
        //writeMatrixCoordinates(data.interpolatedData, data.stratumName);
        //layers.push_back(toPointVector(data.interpolatedData));
    }

    return lithologyMap;
}
void calculatePointDensity(WorkingArea* area, int maxPointCount) {

    int width = static_cast<int>(area->boundingRect.maxX - area->boundingRect.minX);
    int height = static_cast<int>(area->boundingRect.maxY - area->boundingRect.minY);

    int gcd = std::gcd(width, height);
    int aspectWidth = width / gcd;
    int aspectHeight = height / gcd;

    int numRows = static_cast<int>(std::sqrt(maxPointCount / (aspectWidth / static_cast<double>(aspectHeight))));
    int numCols = maxPointCount / numRows;

    if (numRows * numCols > maxPointCount)
        numCols = maxPointCount / numRows;

    area->xScale = (area->boundingRect.maxX - area->boundingRect.minX) / numCols;
    area->yScale = (area->boundingRect.maxY - area->boundingRect.minY) / numRows;
    area->xAxisPoints = numCols;
    area->yAxisPoints = numRows;
    ;
}
