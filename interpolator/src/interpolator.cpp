#include "interpolator.h"
std::vector<Point> toPointVector(const Eigen::MatrixXd& matrix)
{
    std::vector<Point> points;
    points.reserve(matrix.rows() * matrix.cols());

    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            Point point;
            point.x = (double)i;
            point.y = (double)j;
            point.z = matrix(i,j);
            points.push_back(point);
        }
    }

    return points;
}
std::vector<std::vector<Point>> interpolate() {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::map<std::string, LithologyData> lithologyMap;
    std::vector<std::vector<Point>> layers;
    readObservationDataFromJson(lithologyMap, "../../../res/boreholes/borehole_data.json", &maxX, &maxY);
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
        layers.push_back(toPointVector(data.interpolatedData));
    }

    return layers;
}
