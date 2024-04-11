#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <filesystem>
#include <eigen3/Eigen/Dense>
#include "models.h"
#include "kriging_utilities.cuh"

void gnuPlotMatrix(const Eigen::MatrixXd& matrix, std::string formation, std::vector<DataPoint>* data, int maxX, int maxY);

void gnuPlotVariogram(std::string formation, EmpiricalVariogram* vari, TheoreticalParam param);

bool CreateDirectoryRecursive(std::string const& dirName, std::error_code& err);