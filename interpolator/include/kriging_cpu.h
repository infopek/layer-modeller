
#pragma once
#include <eigen3/Eigen/Dense>
#include <models/point.h>
#include <models/lithology_data.h>
#include "models.h"
#include <iostream>
#include <variogram.h>
#include <plotting.h>

class KrigingCalculator{
    public:
        KrigingCalculator();
        ~KrigingCalculator();
        void createInterpolation(LithologyData &lithoData, WorkingArea &area, bool useRegularization=true);
        void crossValidateInterpolation(LithologyData& lithoData, WorkingArea& area,bool useRegularization);
    private:
        void createVariogram(LithologyData& lithoData);
        Eigen::MatrixXd calculateCovarianceMatrix(std::vector<Point>& observedData, TheoreticalParam param);
        KrigingOutput krigingForPoint(LithologyData& lithoData, const Eigen::FullPivLU<Eigen::MatrixXd>& luCovMatrix, double targetX, double targetY);
        double computeConditionNumber(const Eigen::MatrixXd& R);
        Eigen::MatrixXd ridgeRegression(const Eigen::MatrixXd& R, double kmax);
        Eigen::FullPivLU<Eigen::MatrixXd> createCovMatrix(LithologyData& lithoData, bool useRegularization);
};
