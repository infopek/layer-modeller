#include "initialization.h"

void readObservationDataFromJson(std::map<std::string, LithologyData> &lithologyMap, std::string path)
{
    std::ifstream f(path);
    json dataJson = json::parse(f);

    for (const auto &entry : dataJson)
    {
        Point point;
        std::string lithoType = entry["reteg$lito$geo"];
        point.x = static_cast<double>(entry["eovX"]);
        point.y = static_cast<double>(entry["eovY"]);
        point.z = entry["reteg$mig"].get<double>();
        lithologyMap[lithoType].points.push_back(point);
        if (lithologyMap[lithoType].stratumName.empty())
            lithologyMap[lithoType].stratumName = entry["reteg$lito$nev"];
    }



}
