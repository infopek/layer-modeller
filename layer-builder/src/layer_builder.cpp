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
    : m_numLayers{ 2 }
{
    m_layers.resize(m_numLayers);
    m_layers[0].points = points;
    m_layers[1].points.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        m_layers[1].points[i] = Point{ points[i].x + 30.0, points[i].y, points[i].z + 40.0 };
    }

}

LayerBuilder::~LayerBuilder()
{

}

void LayerBuilder::buildLayers()
{
    // std::vector<std::vector<Point>> allPoints = kriging.interpolate(m_regionName);  // 
    // finalize(allPoints);
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
