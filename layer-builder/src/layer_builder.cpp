#include <layer_builder.h>
#include <interpolator.h>
#include <blur/blur.h>
#include <geotiff_handler.h>
#include <normalizer.h>

#include <algorithm>
#include <random>

LayerBuilder::LayerBuilder(const std::string& regionName, const std::string& observationDataPath, const std::string& tiffPath)
    : m_regionName{ regionName }, m_observationDataPath{ observationDataPath }, m_tiffPath{ tiffPath }
{

}

LayerBuilder::LayerBuilder(const std::vector<Point>& points)
{
    // m_layers[0].points = points;
    // m_layers[0].composition = "comp1";

    // m_layers[1].points.resize(points.size());
    // m_layers[1].composition = "comp2";
    // for (size_t i = 0; i < points.size(); i++)
    //     m_layers[1].points[i] = Point{ points[i].x, points[i].y, (points[i].z + rand() % 9) + 40.0 };

    // m_layers[2].points.resize(points.size());
    // m_layers[2].composition = "comp3";
    // for (size_t i = 0; i < points.size(); i++)
    //     m_layers[2].points[i] = Point{ points[i].x, points[i].y, (points[i].z + rand() % 9) + 70.0 };

    // m_layers[3].points.resize(points.size());
    // m_layers[3].composition = "comp4";
    // for (size_t i = 0; i < points.size(); i++)
    //     m_layers[3].points[i] = Point{ points[i].x, points[i].y, (points[i].z + rand() % 9) + 110.0 };

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
        auto& data = it->second;
        m_layers[i].points.resize(data.interpolatedData.size());
        m_layers[i].composition = "comp" + std::to_string(i);

        std::copy(data.interpolatedData.begin(), data.interpolatedData.end(), m_layers[i].points.begin());
        i++;
    }
}