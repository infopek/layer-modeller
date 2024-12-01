#include <layer_builder.h>
#include <interpolator.h>
#include <blur/blur.h>
#include <geotiff_handler.h>
#include <normalizer.h>
#include <client.h>
#include <zip-handler.h>
#include <algorithm>
#include <random>

std::string LayerBuilder::s_logPrefix = "[LAYER_BUILDER] --";

LayerBuilder::LayerBuilder(const std::string &location, const std::string &observationDataPath, const std::string &tiffPath)
    : m_regionName{location},
      m_observationDataPath{observationDataPath},
      m_tiffPath{tiffPath}
{
    #ifndef EVALUATION_MODE_ENABLED
    if (!location.empty())
    {
        std::string resPath = "./res";
        MemoryStruct zipData;
        if (!downloadZipFile(location, zipData))
        {
            throw std::runtime_error("Failed to download ZIP file for region: " + location);
        }
        if (!std::filesystem::exists(resPath))
        {
            std::filesystem::create_directory(resPath);
        }
        std::string geotiffFileName, jsonFileName;
        if (!extractZipFromMemory(zipData.memory, resPath))
        {
            throw std::runtime_error("Failed to extract ZIP file for region: " + location);
        }
        m_tiffPath = resPath + "/raster.tif";
        m_observationDataPath = resPath + "/boreholes.json";
    }
    #endif
}

LayerBuilder::~LayerBuilder()
{
}

void LayerBuilder::buildLayers()
{
    WorkingArea area;

#ifdef EVALUATION_MODE_ENABLED
    BoundingRectangle bTest;
    bTest.maxX=100;
    bTest.maxY=100;
    bTest.minX=0;
    bTest.minY=0;
    area.boundingRect=bTest;
#else
    GeoTiffHandler geoTiff(m_tiffPath);
    area.boundingRect = geoTiff.getBoundingRectangle();
#endif
    Interpolator interpolator;
    std::vector<std::pair<std::string, LithologyData>> allLayers = interpolator.interpolate(area, m_observationDataPath);
#ifndef EVALUATION_MODE_ENABLED    
    normalizeLayers(allLayers, geoTiff, area);
    layerize(allLayers);
#endif

}

void LayerBuilder::layerize(std::vector<std::pair<std::string, LithologyData>> &layers)
{
    m_numLayers = layers.size();
    m_layers.resize(m_numLayers);

    int i = 0;
    for (auto it = layers.begin(); it != layers.end(); ++it)
    {
        Logger::log(LogLevel::INFO, LayerBuilder::s_logPrefix + " Processing layer " + std::to_string(i + 1) + "...");

        auto &data = it->second;
        m_layers[i].points.resize(data.interpolatedData.size());
        m_layers[i].composition = "comp" + std::to_string(i);

        std::copy(data.interpolatedData.begin(), data.interpolatedData.end(), m_layers[i].points.begin());
        i++;
    }
}