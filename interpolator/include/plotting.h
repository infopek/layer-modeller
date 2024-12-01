#pragma once
#pragma execution_character_set("utf-8")
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include "kriging_utilities.cuh"
#include <models/point.h>
#include <models/lithology_data.h>
#include <iomanip>

void gnuPlotTestValidation(const std::map<std::string, CalculationRunTime>& dataMap, const std::string& plotTitle);

void gnuPlotArea(std::vector<Point> data, std::string stratumName, const WorkingArea& area, std::string dataTypeStr,std::string properties="");

void gnuPlotValidity(const LithologyData& lithodata, const WorkingArea& area,std::vector<Point> data,std::string properties);

void gnuPlotKriging(const LithologyData& lithodata, const WorkingArea& area,std::string properties);

void gnuPlotMatrix(std::string name, const Eigen::MatrixXd& matrix, std::string formation, std::vector<Point>* data, int maxX, int maxY);

void gnuPlotVariogram(LithologyData& lithoData);

bool createDirectoryRecursive(std::string const& dirName, std::error_code& err);

void writeMatrixCoordinates(const Eigen::MatrixXd& matrix, const std::string& formation);

void gnuPlotValidationMatrix(const std::map<std::pair<std::string,std::string>,double>& mx );
std::string getDir(const std::string& formation);
