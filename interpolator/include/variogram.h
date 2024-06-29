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
//__global__ void calculateValues(int n, float* D, float* S, const DataPoint* points);

static void calculateValuesCPU(int n, double* D, double* S, std::vector<Point>  points);

void createVariogram(std::vector<Point>* points, EmpiricalVariogram* variogramData);

void callback(const size_t iter, void* params, const gsl_multifit_nlinear_workspace* w);

int gaussianModel(const gsl_vector* x, void* data, gsl_vector* f);

TheoreticalParam fitTheoreticalFunction(EmpiricalVariogram* variogram);