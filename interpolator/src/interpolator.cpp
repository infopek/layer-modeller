#include "interpolator.h"

#include <string>
#include <initialization.h>

std::vector<std::pair<std::string, LithologyData>> Interpolator::interpolate(WorkingArea &area, const std::string &observationDataPath)
{
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::vector<std::pair<std::string, LithologyData>> lithologyVector;
    calculatePointDensity(area, 10000);
#ifdef EVALUATION_MODE_ENABLED
    std::vector<Point> testLayer;
    generateTestData(lithologyVector, testLayer);
    gnuPlotArea(testLayer,"TEST-LAYER",area,"test-layer");
#else
    readObservationDataFromJson(lithologyVector, observationDataPath);
#endif
    KrigingCalculator kriging;
    for (auto it = lithologyVector.begin(); it != lithologyVector.end(); ++it)
    {
        auto &data = it->second;
        std::cout << it->first << std::endl;
        kriging.createInterpolation(data, area, false);
    }
#ifdef EVALUATION_MODE_ENABLED
    validateAndGroup(lithologyVector, testLayer);
#endif
    return lithologyVector;
}
void Interpolator::calculatePointDensity(WorkingArea &area, int maxPointCount)
{

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
double Interpolator::calculateAverage(const std::vector<double>& vec) {
    if (vec.empty()) return 0.0;
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    return sum / vec.size();
}
CalculationRunTime Interpolator::calculateAverage(const std::vector<CalculationRunTime>& vec) {
    CalculationRunTime average = {0, 0, 0};
    
    if (vec.empty()) {
        return average;
    }

    for (const auto& item : vec) {
        average.variogram += item.variogram;
        average.covMatrix += item.covMatrix;
        average.kriging += item.kriging;
    }

    long long size = vec.size();
    average.variogram /= size;
    average.covMatrix /= size;
    average.kriging /= size;

    return average;
}
void Interpolator::validateAndGroup(
    const std::vector<std::pair<std::string, LithologyData>> &lithologyVector,
    const std::vector<Point> &testLayer)
{
    // Maps for grouping by number of observed points and deviation scale
    std::map<std::pair<std::string,std::string>, double> validationMatrix;
    std::map<std::string, std::vector<CalculationRunTime>> runTimeGroups;
    std::map<std::string, CalculationRunTime> runTimes;


    for (const auto &[layerName, lithologyData] : lithologyVector)
    {
        std::stringstream ss(layerName);
        std::string observedPoints, deviationScale;
        getline(ss, observedPoints, '_');
        getline(ss, deviationScale, '_');

        // Perform comparison
        std::vector<Point> interpolatedPoints = lithologyData.interpolatedData;
        std::vector<double> zDifferences;
        for (size_t i = 0; i < interpolatedPoints.size(); ++i)
        {
            const Point &interpolatedPoint = interpolatedPoints[i];
            const Point &testPoint = testLayer[i]; // Assuming the same order and size
            zDifferences.push_back(std::abs(interpolatedPoint.z - testPoint.z));
            
        }
        double avg=calculateAverage(zDifferences);
        validationMatrix[{observedPoints,deviationScale}]=avg;
        runTimeGroups[observedPoints].push_back(lithologyData.runTimes);
    }
    for(auto group:runTimeGroups){
        runTimes[group.first]= calculateAverage(group.second);
    }

    gnuPlotValidationMatrix(validationMatrix);
    gnuPlotTestValidation(runTimes, "Run times");
}