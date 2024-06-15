#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include <iostream>
#include <string>
#include <iostream>
#include <tuple>

using json = nlohmann::json;

void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap, std::string path, int* maxX, int* maxY);
std::string replaceAccents(const std::string& input);
void readTiff();
//void readObservationDataFromJson(std::map<std::string, LithologyData>& lithologyMap,std::string path, int* maxX, int* maxY, double* scaleFactor);
//std::string replaceAccents(const std::string& input);
//void readTiff(int& minX, int& minY, int& maxX, int& maxY);
