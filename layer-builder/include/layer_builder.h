#pragma once

#include <models/point.h>
#include <models/layer.h>
#include <models/lithology_data.h>
#include <logging.h>

#include <string>
#include <vector>
#include <map>

class LayerBuilder
{
public:
    LayerBuilder(const std::string& regionName, const std::string& observationDataPath, const std::string& tiffPath);
    ~LayerBuilder();

    inline size_t getNumLayers() const { return m_numLayers; }
    inline const std::vector<Layer>& getLayers() const { return m_layers; }

    void buildLayers();

private:
    void layerize(std::vector<std::pair<std::string, LithologyData>>& layers);

private:
    std::vector<Layer> m_layers{};
    size_t m_numLayers{};

    std::string m_regionName{};
    std::string m_observationDataPath{};
    std::string m_tiffPath{};

    static std::string s_logPrefix;
};
