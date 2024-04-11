#pragma once
#include <fstream>
#include <nlohmann/json.hpp>

#include "models.h"

using json = nlohmann::json;

void readObservationDataFromJson(std::vector<DataPoint>* data, std::string path, std::string formation, int* maxX, int* maxY);