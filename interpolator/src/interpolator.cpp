#include "interpolator.h"

std::map<std::string, LithologyData> interpolate() {
    std::locale::global(std::locale("en_US.UTF-8"));
    int maxX = 0, maxY = 0;
    std::map<std::string, LithologyData> lithologyMap;
    BoundingRectangle boundingRect;
    readObservationDataFromJson(lithologyMap, "../../../res/boreholes/borehole_kovago.json", &boundingRect);
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& lithoType = it->first;
        auto& data = it->second;
        EmpiricalVariogram empiricalData;
      
        std::cout << lithoType << std::endl << std::endl;
        createVariogram(&data.points, &empiricalData);

        data.theoreticalParam = fitTheoreticalFunction(&empiricalData);
        
        //createInterpolationGPU(&data, param, &krigingData, maxX, maxY, plotSize);

        createInterpolation(&data.points, &data, &boundingRect);

        // gnuPlotMatrix("values", data.interpolatedData, data.stratumName, &data.points, maxX, maxY);
        // gnuPlotMatrix("certainty", data.certaintyMatrix, data.stratumName, &data.points, maxX, maxY);
        gnuPlotVariogram(data.stratumName, &empiricalData, data.theoreticalParam);
        //writeMatrixCoordinates(data.interpolatedData, data.stratumName);
        //layers.push_back(toPointVector(data.interpolatedData));
    }
    std::vector<std::string> seq = {"Qh","jT1", "k_c-tP3-T1",  "k_kP3"};
        seq.push_back("Qh");
    seq.push_back("k_c-tP3-T1");
    seq.push_back("jT1");
    seq.push_back("k_kP3");
   for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& data = it->second;
        auto place = std::find(seq.begin(), seq.end(), it->first);
        if (place != seq.end() && place != seq.begin()) {
            // Find the previous element in seq
            auto prevPlace = std::prev(place);
            std::string prevKey = *prevPlace;

            if (lithologyMap.find(prevKey) != lithologyMap.end()) {
                auto& prevData = lithologyMap[prevKey];

                // Iterate through data.interpolatedData
                for (size_t i = 0; i < data.interpolatedData.size(); ++i) {
                    // Ensure there's a corresponding element in prevData.interpolatedData
                    if (i < prevData.interpolatedData.size()) {
                        if (prevData.interpolatedData[i].z > data.interpolatedData[i].z) {
                            data.interpolatedData[i].z = prevData.interpolatedData[i].z;
                        }
                    }
                }
            }
        }
    }
    return lithologyMap;
}
