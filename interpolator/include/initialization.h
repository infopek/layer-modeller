#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include <iostream>
#include <string>
#include <iostream>
#include <tuple>
#include <models/point.h>
#include <models/lithology_data.h>

using json = nlohmann::json;
void filterMaxElements(std::unordered_map<std::string, std::vector<Point>>& map) ;

void readObservationDataFromJson(std::vector<std::pair<std::string, LithologyData>> &lithologyVector, const std::string &path);
//std::string replaceAccents(const std::string& input);
// void readTiff();
//void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap,std::string path, int* maxX, int* maxY, double* scaleFactor);
//std::string replaceAccents(const std::string& input);
//void readTiff(int& minX, int& minY, int& maxX, int& maxY);
