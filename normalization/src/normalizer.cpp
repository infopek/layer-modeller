#include <normalizer.h>
#include <iostream>

void resolveCrossingLayers(std::vector<std::pair<std::string, LithologyData>> &lithologyVector)
{
    for (auto it = lithologyVector.begin() + 1; it != lithologyVector.end(); ++it)
    {

        auto &data = it->second;
        auto prevData = (*std::prev(it)).second;
        for (size_t i = 0; i < data.interpolatedData.size(); ++i)
            if (i < prevData.interpolatedData.size())
                if (prevData.interpolatedData[i].z < data.interpolatedData[i].z)
                    data.interpolatedData[i].z = prevData.interpolatedData[i].z;
    }
}
std::vector<Point> getFactorOfPreviosLayer(float strength, std::vector<Point> certainty)
{
    std::vector<Point> scaledCertainty;

    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();

    for (const auto& point : certainty) {
        if(point.z<minZ) minZ=point.z;
        if(point.z>maxZ) maxZ=point.z;
    }
    float zeroFactor=minZ+(maxZ-minZ)*strength;
    float maxFactor=maxZ-zeroFactor;
    for (const auto& point : certainty) {
        float factor = point.z-zeroFactor;
        if(factor<0)factor=0;
        scaledCertainty.push_back({.x=point.x,.y=point.y,.z= factor/maxFactor});

    }
    return scaledCertainty;
}
void shiftPointsBasedOnBlur(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler &geoTiff,WorkingArea &area)
{
    float *raster = geoTiff.getRaster();
    geoTiff.blurRaster(raster);
    LithologyData soil;
    for (int i = 0; i < area.yAxisPoints; i++)
    {
        for (int j = 0; j < area.xAxisPoints; j++)
        {
            auto virtualLocation = i * area.xAxisPoints + j;
            double realX = j * area.xScale;
            double realY = i * area.yScale;
            auto realLocation = static_cast<int>(realY) * static_cast<int>(area.xAxisPoints * area.xScale) + static_cast<int>(realX);
            soil.interpolatedData.push_back(Point{
                .x = area.boundingRect.minX + realX,
                .y = area.boundingRect.minY + realY,
                .z = raster[realLocation]});
        }
    }
    LithologyData previous=soil;
    previous.averageDepth=0;
    for (auto it = lithologyVector.begin(); it != lithologyVector.end(); ++it)
    {
        auto &data = it->second;
        auto factors= getFactorOfPreviosLayer(0.7,data.certaintyMatrix);
        gnuPlotArea(factors,data.stratumName,area,"factors");
        auto averageDepthDiff= data.averageDepth-previous.averageDepth;
        std::cout<<averageDepthDiff<<"m "<<std::endl;
        for (int i = 0; i < area.yAxisPoints; i++)
        {
            for (int j = 0; j < area.xAxisPoints; j++)
            {
                auto virtualLocation = i * area.xAxisPoints + j;

                data.interpolatedData[virtualLocation].z = 
                ( previous.interpolatedData[virtualLocation].z-averageDepthDiff)*factors[virtualLocation].z+
                ((soil.interpolatedData[virtualLocation].z-data.interpolatedData[virtualLocation].z)*(1-factors[virtualLocation].z));
            }
        }
        previous=data;
    }
    geoTiff.freeRaster(raster);
    std::pair soilPair("TOP-SOIL", soil);
    lithologyVector.push_back(soilPair);
    std::rotate(lithologyVector.rbegin(), lithologyVector.rbegin() + 1, lithologyVector.rend());
    std::cout << "Sorted reteglitonev names: ";
    for (const auto &name : lithologyVector)
    {
        std::cout << name.first << " ";
    }
    std::cout << std::endl;
}
void normalizeLayers(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, GeoTiffHandler &geoTiff, WorkingArea &area)
{
    shiftPointsBasedOnBlur(lithologyVector, geoTiff, area);
    resolveCrossingLayers(lithologyVector);
}