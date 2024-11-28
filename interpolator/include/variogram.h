#pragma once
#include <functional>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include "kriging_utilities.cuh"
#include <models/point.h>
#include <models/lithology_data.h>
#include <algorithm>
#include <iostream>
#include <models/point.h>
#include "logging.h"
#include <chrono> 
class VariogramCalculator
{
public:
    VariogramCalculator(std::vector<Point> &points);
    ~VariogramCalculator();
    inline const Variogram getVariogram() { return variogram; }
private:
    void computeEmpiricalVariogram(std::vector<Point> &points);
    void fitTheoreticalFunction();
    static void callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w);
    static int gaussianModel(const gsl_vector *x, void *data, gsl_vector *f);

    Variogram variogram;

    static std::string s_logPrefix;
};
