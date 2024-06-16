#include <layer_builder.h>

// #include <normalize.h>
#include <interpolator.h>
#include <blur/blur.h>

#include <algorithm>
#include <random>

LayerBuilder::LayerBuilder(const std::string& regionName)
    : m_regionName{ regionName }
{
}

LayerBuilder::LayerBuilder(const std::vector<Point>& points)
{

    buildLayers();
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
    std::vector<std::vector<Point>> allPoints = interpolate();
    m_numLayers=allPoints.size();
    m_layers.clear();
    m_layers.resize(m_numLayers);

    for(int i = 0; i < m_numLayers; ++i)
    {
        m_layers[i].points.resize(allPoints[i].size());
        m_layers[i].composition = "comp" + std::to_string(i);
        
        for (size_t j = 0; j < allPoints[i].size(); ++j)
        {
            m_layers[i].points[j] = Point{ allPoints[i][j].x, allPoints[i][j].y, allPoints[i][j].z };
            //std::cout << "x: " << m_layers[i].points[j].x << " y: " << m_layers[i].points[j].y << " z: " << m_layers[i].points[j].z << std::endl;
        }
    }
    
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
