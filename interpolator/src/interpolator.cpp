
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "plotting.h"
#include "variogram.cuh"
#include "initialization.h"
#include "kriging_cpu.h"

int main() {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::map<std::string, LithologyData> lithologyMap;
   //readTiff();
    readObservationDataFromJson(lithologyMap, "borehole_kovago.json", &maxX, &maxY);
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        EmpiricalVariogram empiricalData;
        std::cout << lithoType << std::endl << std::endl;
        createVariogram(&data.points, &empiricalData);

        data.theoreticalParam = fitTheoreticalFunction(&empiricalData);
        
        //createInterpolationGPU(&data, param, &krigingData, maxX, maxY, plotSize);

        createInterpolation(&data.points, data.theoreticalParam, &data.interpolatedData, &data.certaintyMatrix, maxX, maxY);

        gnuPlotMatrix("values", data.interpolatedData, data.stratumName, &data.points, maxX, maxY);
        gnuPlotMatrix("certainty", data.certaintyMatrix, data.stratumName, &data.points, maxX, maxY);
        gnuPlotVariogram(data.stratumName, &empiricalData, data.theoreticalParam);
        writeMatrixCoordinates(data.interpolatedData, data.stratumName);
    }

    return 0;
}