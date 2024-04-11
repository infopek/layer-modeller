
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <cmath>
#include "plotting.h"
#include "variogram.cuh"
#include "initialization.h"
#include "kriging_cpu.h"
#include "kriging_gpu.cuh"

int main() {
    int maxX = 0, maxY = 0;
    std::string formation = "Rudolfhazi Homok Retegtag";
    EmpiricalVariogram empiricalData;
    std::vector<DataPoint> data;

    readObservationDataFromJson(&data, "../../../res/boreholes/borehole_data.json", formation, &maxX, &maxY);
    createVariogram(&data, &empiricalData);

    TheoreticalParam param = fitTheoreticalFunction(&empiricalData);

    const int plotSize = 128;
    double* krigingData = new double[plotSize * plotSize];

    //createInterpolationGPU(&data, param, &krigingData, maxX, maxY, plotSize);
    Eigen::MatrixXd krigingMx = Eigen::MatrixXd::Map(krigingData, plotSize, plotSize);

    createInterpolation(&data, param, &krigingMx, maxX, maxY);

    gnuPlotMatrix(krigingMx, formation, &data, maxX, maxY);
    gnuPlotVariogram(formation, &empiricalData, param);

    return 0;
}