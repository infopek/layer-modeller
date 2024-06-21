#include "layer_normalizer.h"

void resolveCrossingLayers(std::map<std::string, LithologyData>& lithologyMap){
    std::vector<std::string> seq = {"Qh","jT1", "k_c-tP3-T1",  "k_kP3"};//{"talaj","kisberi","csatkai","szoci","rudolfhazi","kisgyoni","ganti"};//
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
        auto& data = it->second;
        auto place = std::find(seq.begin(), seq.end(), it->first);
        if (place != seq.end() && place != seq.begin()) {
            auto prevPlace = std::prev(place);
            std::string prevKey = *prevPlace;

            if (lithologyMap.find(prevKey) != lithologyMap.end()) {
                auto& prevData = lithologyMap[prevKey];
                for (size_t i = 0; i < data.interpolatedData.size(); ++i) {
                    if (i < prevData.interpolatedData.size()) {
                        if (prevData.interpolatedData[i].z > data.interpolatedData[i].z) {
                            data.interpolatedData[i].z = prevData.interpolatedData[i].z;
                        }
                    }
                }
            }
        }
    }
}
void shiftPointsBasedOnBlur(std::map<std::string, LithologyData>& lithologyMap){
    
}
void normalize(std::map<std::string, LithologyData>& lithologyMap){
    shiftPointsBasedOnBlur(lithologyMap);
    resolveCrossingLayers(lithologyMap);
}