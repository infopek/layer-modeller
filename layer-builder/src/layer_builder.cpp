#include <layer_builder.h>

// #include <normalize.h>
#include <kriging_cpu.h>
#include <blur/blur.h>

#include <algorithm>

LayerBuilder::LayerBuilder(const std::string& regionName)
    : m_regionName{ regionName }
{
}

LayerBuilder::LayerBuilder(const std::vector<Point>& points)
    : m_numLayers{ 1 }
{
    m_layers.resize(m_numLayers);
    m_layers[0].points = points;
}

LayerBuilder::~LayerBuilder()
{

}

void LayerBuilder::buildLayers()
{
    // std::vector<std::vector<Point>> allPoints = kriging.interpolate(m_regionName);  // 
    // normalizer.normalize(allPoints);    // normalize
    // layerize(allPoints);
}

void LayerBuilder::layerize(const std::vector<std::vector<Point>>& allPoints)
{
    std::for_each(allPoints.cbegin(), allPoints.cend(),
        [&](const std::vector<Point>& points)
        {
            Layer layer{};
            layer.points = points;
            //layer.mineralMap = <where from>
            //...

            m_layers.push_back(layer);
            ++m_numLayers;
        });
}
