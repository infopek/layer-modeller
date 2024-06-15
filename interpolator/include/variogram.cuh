
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
#include "cuda.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
//__global__ void calculateValues(int n, float* D, float* S, const DataPoint* points);

static void calculateValuesCPU(int n,  double* D, double* S, std::vector<DataPoint>  points);

void createVariogram(std::vector<DataPoint>* points, EmpiricalVariogram* variogramData);

void callback(const size_t iter, void* params, const gsl_multifit_nlinear_workspace* w);

int gaussianModel(const gsl_vector* x, void* data, gsl_vector* f);

TheoreticalParam fitTheoreticalFunction(EmpiricalVariogram* variogram);