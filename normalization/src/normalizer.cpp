#include <normalizer.h>
#include <iostream>

void resolveCrossingLayers(std::map<std::string, LithologyData>& lithologyMap) {
    std::vector<std::string> seq = { "Qh","jT1", "k_c-tP3-T1",  "k_kP3" };//{"talaj","kisberi","csatkai","szoci","rudolfhazi","kisgyoni","ganti"};//
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
void shiftPointsBasedOnBlur(std::map<std::string, LithologyData>& lithologyMap,GeoTiffHandler* geoTiff, WorkingArea* area) {
    float* raster = geoTiff->getRaster();
    LithologyData soil;
        //     std::cout<<"size: "<<interploated.size()<<" calculated: "<<area->xAxisPoints*area->yAxisPoints<<std::endl;
        // std::cout<<std::endl;
    for(int i=0; i<area->yAxisPoints; i++){
        for(int j=0; j<area->xAxisPoints; j++){
            for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it) {
                auto& data = it->second;
                auto virtualLocation=i*area->yAxisPoints+j;
                auto realLocation= static_cast<int>(i*area->yAxisPoints*area->xScale+j*area->yScale);

                data.interpolatedData[virtualLocation].z=raster[realLocation]-data.interpolatedData[virtualLocation].z;

            }
        }
    }

}
void normalizeLayers(std::map<std::string, LithologyData>& lithologyMap,GeoTiffHandler* geoTiff, WorkingArea* area) {
    shiftPointsBasedOnBlur(lithologyMap,geoTiff,area);
    resolveCrossingLayers(lithologyMap);
}