#pragma once

#include <models/point.h>
#include <models/layer.h>
#include <models/lithology_data.h>

#include <string>
#include <vector>
#include <map>

class LayerBuilder
{
public:
    // To be deleted
    LayerBuilder(const std::vector<Point>& points);

    LayerBuilder(const std::string& regionName, const std::string& observationDataPath, const std::string& tiffPath);
    ~LayerBuilder();

    inline size_t getNumLayers() const { return m_numLayers; }
    inline const std::vector<Layer>& getLayers() const { return m_layers; }

    void buildLayers();

private:
    void layerize(const std::map<std::string, LithologyData>& layers);

private:
    std::vector<Layer> m_layers{};
    size_t m_numLayers{};

    std::string m_regionName{};
    std::string m_observationDataPath{};
    std::string m_tiffPath{};
};
