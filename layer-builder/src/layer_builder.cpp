#include <layer_builder.h>
#include <interpolator.h>
#include <blur/blur.h>
#include <geotiff_handler.h>
#include <normalizer.h>

#include <algorithm>
#include <random>

std::string LayerBuilder::s_logPrefix = "[LAYER_BUILDER] --";

LayerBuilder::LayerBuilder(const std::string& regionName, const std::string& observationDataPath, const std::string& tiffPath)
    : m_regionName{ regionName },
    m_observationDataPath{ observationDataPath },
    m_tiffPath{ tiffPath }
{

}

LayerBuilder::~LayerBuilder()
{

}

void LayerBuilder::buildLayers()
{
    WorkingArea area;
    GeoTiffHandler geoTiff(m_tiffPath);
    area.boundingRect = geoTiff.getBoundingRectangle();

    std::map<std::string, LithologyData> allLayers = interpolate(&area, m_observationDataPath);

    normalizeLayers(allLayers, &geoTiff, &area);
    layerize(allLayers);
}

void LayerBuilder::layerize(const std::map<std::string, LithologyData>& layers)
{
    m_numLayers = layers.size();
    m_layers.resize(m_numLayers);

    int i = 0;
    for (auto it = layers.begin(); it != layers.end(); ++it)
    {
        Logger::log(LogLevel::INFO, LayerBuilder::s_logPrefix + " Processing layer " + std::to_string(i + 1) + "...");

        auto& data = it->second;
        m_layers[i].points.resize(data.interpolatedData.size());
        m_layers[i].composition = "comp" + std::to_string(i);

        std::copy(data.interpolatedData.begin(), data.interpolatedData.end(), m_layers[i].points.begin());
        i++;
    }
}