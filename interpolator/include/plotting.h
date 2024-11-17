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
#include <models/lithology_data.h>

void gnuPlotValidity(const LithologyData& lithodata, const WorkingArea& area,std::vector<Point> data);

void gnuPlotKriging(const LithologyData& lithodata, const WorkingArea& area);

void gnuPlotMatrix(std::string name, const Eigen::MatrixXd& matrix, std::string formation, std::vector<Point>* data, int maxX, int maxY);

void gnuPlotVariogram(LithologyData& lithoData, EmpiricalVariogram* vari);

bool createDirectoryRecursive(std::string const& dirName, std::error_code& err);

void writeMatrixCoordinates(const Eigen::MatrixXd& matrix, const std::string& formation);

std::string getDir(const std::string& formation);