#include <normalizer.h>
#include <iostream>

void resolveCrossingLayers(std::vector<std::pair<std::string, LithologyData>> &lithologyVector)
{
    for (auto it = lithologyVector.begin()+1; it != lithologyVector.end(); ++it)
    {

        auto &data = it->second;
        auto prevData = (*std::prev(it)).second;
        for (size_t i = 0; i < data.interpolatedData.size(); ++i)
        {
            if (i < prevData.interpolatedData.size())
            {
                if (prevData.interpolatedData[i].z < data.interpolatedData[i].z)
                {
                    data.interpolatedData[i].z = prevData.interpolatedData[i].z;
                }
            }
        }
    }
}
void shiftPointsBasedOnBlur(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler *geoTiff, WorkingArea *area)
{
    float *raster = geoTiff->getRaster();
    LithologyData soil;

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
                .z = raster[realLocation]});
            for (auto it = lithologyVector.begin(); it != lithologyVector.end(); ++it)
            {
                auto &data = it->second;
                // std::cout<<"depth: "<<data.interpolatedData[virtualLocation].z<<" lidar: "<<raster[realLocation]<<" calculated: "<<raster[realLocation]-data.interpolatedData[virtualLocation].z<<std::endl;
                data.interpolatedData[virtualLocation].z = raster[realLocation] - data.interpolatedData[virtualLocation].z;
            }
        }
    }
    geoTiff->freeRaster(raster);
    std::pair soilPair("TOP-SOIL",soil);
    lithologyVector.push_back(soilPair);
    std::rotate(lithologyVector.rbegin(), lithologyVector.rbegin() + 1, lithologyVector.rend());
    std::cout << "Sorted reteglitonev names: ";
    for (const auto &name : lithologyVector)
    {
        std::cout << name.first << " ";
    }
    std::cout << std::endl;

}
void normalizeLayers(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler *geoTiff, WorkingArea *area)
{
    shiftPointsBasedOnBlur(lithologyVector, geoTiff, area);
    resolveCrossingLayers(lithologyVector);
}