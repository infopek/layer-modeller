#pragma once

#include <models/point.h>
#include <models/layer.h>

#include <string>
#include <vector>

class LayerBuilder
{
public:
    // To be deleted
    LayerBuilder(const std::vector<Point>& points);

    LayerBuilder(const std::string& regionName);
    ~LayerBuilder();

    inline size_t getNumLayers() const { return m_numLayers; }
    inline const std::vector<Layer>& getLayers() const { return m_layers; }

    void buildLayers();

private:
    void layerize(const std::vector<std::vector<Point>>& allPoints);

private:
    std::vector<Layer> m_layers{};
    size_t m_numLayers{};

    std::string m_regionName{};
};
