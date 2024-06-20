#pragma once
#pragma execution_character_set("utf-8")
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include "kriging_utilities.cuh"
#include <models/point.h>
#include <models/lithologyData.h>

void gnuPlotMatrix(std::string name ,const Eigen::MatrixXd& matrix, std::string formation, std::vector<Point>* data, int maxX, int maxY);

void gnuPlotVariogram(std::string formation, EmpiricalVariogram* vari, TheoreticalParam param);

bool createDirectoryRecursive(std::string const& dirName, std::error_code& err);

void writeMatrixCoordinates(const Eigen::MatrixXd& matrix, const std::string& formation);

std::string getDir(const std::string& formation);