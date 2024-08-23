#include <normalizer.h>
#include <iostream>

void resolveCrossingLayers(std::map<std::string, LithologyData>& lithologyMap)
{
    std::vector<std::string> seq = { "TOP-SOIL", "Qh", "jT1", "k_c-tP3-T1", "k_kP3" }; //{"talaj","kisberi","csatkai","szoci","rudolfhazi","kisgyoni","ganti"};//
    for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it)
    {
        auto& data = it->second;
        auto place = std::find(seq.begin(), seq.end(), it->first);
        if (place != seq.end() && place != seq.begin())
        {
            auto prevPlace = std::prev(place);
            std::string prevKey = *prevPlace;

            if (lithologyMap.find(prevKey) != lithologyMap.end())
            {
                auto& prevData = lithologyMap[prevKey];
                for (size_t i = 0; i < data.interpolatedData.size(); ++i)
                {
                    if (i < prevData.interpolatedData.size())
                    {
                        if (prevData.interpolatedData[i].z > data.interpolatedData[i].z)
                        {
                            data.interpolatedData[i].z = prevData.interpolatedData[i].z;
                        }
                    }
                }
            }
        }
    }
}
void shiftPointsBasedOnBlur(std::map<std::string, LithologyData>& lithologyMap, GeoTiffHandler* geoTiff, WorkingArea* area)
{
    float* raster = geoTiff->getRaster();
    LithologyData soil;
    // for(int i=0; i<static_cast<int>(area->yAxisPoints*area->yScale); i++){
    //     for(int j=0; j<static_cast<int>(area->xAxisPoints*area->xScale); j++){
    //         auto virtualLocation=i*area->yAxisPoints+j;
    //         auto realLocation= static_cast<int>(i*static_cast<int>(area->xAxisPoints*area->xScale)+j);
    //         soil.interpolatedData.push_back(Point{
    //             .x = area->boundingRect.minX + j,
    //             .y = area->boundingRect.minY + i,
    //             .z = raster[realLocation]
    //         });
    //     }
    // }
    for (int i = 0; i < area->yAxisPoints; i++)
    {
        for (int j = 0; j < area->xAxisPoints; j++)
        {
            auto virtualLocation = i * area->xAxisPoints + j;
            double realX = j * area->xScale;
            double realY = i * area->yScale;
            auto realLocation = static_cast<int>(realY) * static_cast<int>(area->xAxisPoints * area->xScale) + static_cast<int>(realX);
            soil.interpolatedData.push_back(Point{
                .x = area->boundingRect.minX + realX,
                .y = area->boundingRect.minY + realY,
                .z = raster[realLocation] });
            for (auto it = lithologyMap.begin(); it != lithologyMap.end(); ++it)
            {
                auto& data = it->second;
                // std::cout<<"depth: "<<data.interpolatedData[virtualLocation].z<<" lidar: "<<raster[realLocation]<<" calculated: "<<raster[realLocation]-data.interpolatedData[virtualLocation].z<<std::endl;
                data.interpolatedData[virtualLocation].z = raster[realLocation] - data.interpolatedData[virtualLocation].z;
            }
        }
    }
    geoTiff->freeRaster(raster);
    lithologyMap["TOP-SOIL"] = soil;
    ;
}
void normalizeLayers(std::map<std::string, LithologyData>& lithologyMap, GeoTiffHandler* geoTiff, WorkingArea* area)
{
    shiftPointsBasedOnBlur(lithologyMap, geoTiff, area);
    //resolveCrossingLayers(lithologyMap);
}