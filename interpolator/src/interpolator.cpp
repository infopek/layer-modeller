#include "interpolator.h"

std::map<std::string, LithologyData> interpolate() {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::map<std::string, LithologyData> lithologyMap;
    InterpolatedArea area;
    readObservationDataFromJson(lithologyMap, "../../../res/boreholes/borehole_kovago.json", &area);
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        EmpiricalVariogram empiricalData;
      
        std::cout << lithoType << std::endl << std::endl;
        createVariogram(&data.points, &empiricalData);

        data.theoreticalParam = fitTheoreticalFunction(&empiricalData);
        
        //createInterpolationGPU(&data, param, &krigingData, maxX, maxY, plotSize);

        createInterpolation(&data.points, &data, &area);

        // gnuPlotMatrix("values", data.interpolatedData, data.stratumName, &data.points, maxX, maxY);
        // gnuPlotMatrix("certainty", data.certaintyMatrix, data.stratumName, &data.points, maxX, maxY);
        gnuPlotVariogram(data.stratumName, &empiricalData, data.theoreticalParam);
        //writeMatrixCoordinates(data.interpolatedData, data.stratumName);
        //layers.push_back(toPointVector(data.interpolatedData));
    }

    return lithologyMap;
}
