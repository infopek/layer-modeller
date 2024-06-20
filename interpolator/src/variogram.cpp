#include "variogram.h"
#include <algorithm>
#include <iostream>
#include <models/point.h>

void calculateValuesCPU(int n, double* D, double* S, std::vector<Point> points) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {  // compare every point with every other point
            double dx = (double)(points[i].x - points[j].x);
            double dy = (double)(points[i].y - points[j].y);
            D[i * n + j] = std::sqrt(dx * dx + dy * dy);  // distance calculation
            S[i * n + j] = 0.5 * std::pow(points[i].z - points[j].z, 2);  // squared difference
        }
    }
}
//__global__ void calculateValues(int n, uint32_t* k, double* D, double* S, const DataPoint* points) {
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    int i = idx / n;
//    int j = idx % n;
//    if (i < n && j < n && i < j) {
//        uint32_t index = atomicAdd(k, 1);
//        double dx = (double)(points[i].x - points[j].x);
//        double dy = (double)(points[i].y - points[j].y);
//        D[index] = std::sqrt(dx * dx + dy * dy);
//        S[index] = 0.5 * std::pow(points[i].z - points[j].z, 2);
//    }
//}

void createVariogram(std::vector<Point>* points, EmpiricalVariogram* variogramData) { 
    int n = points->size();
    int N = n * (n + 1) / 2;
    std::vector<int> I(N);
    for (int i = 0; i < N; i++) {
        I[i] = i;
    }
    //double* d_D, * d_S;
    //DataPoint *d_points;
    //cudaMalloc((void**)&d_D, sizeof(double) * n * n);
    //cudaMalloc((void**)&d_S, sizeof(double) * n * n);
    //cudaMalloc((void**)&d_points, sizeof(DataPoint) * n);
    //    
    //cudaMemcpy(d_points, &(*points)[0], sizeof(DataPoint) * n, cudaMemcpyHostToDevice);
    //uint32_t* d_k;
    //uint32_t k=0;
    //cudaMalloc((void**)&d_k, sizeof(uint32_t));
    //cudaMemcpy(d_k,&k,sizeof(uint32_t),cudaMemcpyHostToDevice);
    //calculateValues << < (int)(pow(n,2)+511)/512,512>> > (n,d_k, d_D, d_S,d_points );
    //cudaError_t cudaError = cudaGetLastError();
    //if (cudaError != cudaSuccess) {
    //    fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaError));
    //    exit(EXIT_FAILURE);
    //}

    double* D = new double[n*n];
    double* S = new double[n * n];
    //cudaDeviceSynchronize();
    //cudaFree(d_k);
    //cudaMemcpy(D, d_D, sizeof(double) * n * n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(S, d_S, sizeof(double) * n * n, cudaMemcpyDeviceToHost);

    //cudaFree(d_D);
    //cudaFree(d_S);
    //cudaFree(d_points);
    calculateValuesCPU(n, D, S, *points);

    std::sort(I.begin(), I.end(),
        [&](const int& k, const int& l) {
            return (D[k] < D[l]);
        });

    std::vector<double> valvec(N);
    std::vector<double> distvec(N);

    valvec[0]=S[I[0]];
    for (int k = 1; k < N; k++) {
        valvec[k] = (valvec[k - 1] * k + S[I[k]]) / (k + 1.0);
        valvec[k - 1] *= 0.5;
        distvec[k] = D[I[k]];
    }
    valvec[N - 1] *= 0.5;
    variogramData->values = valvec;
    variogramData->distances = distvec;

}
void callback(const size_t iter, void* params, const gsl_multifit_nlinear_workspace* w)
{
    gsl_vector* f = gsl_multifit_nlinear_residual(w);
    gsl_vector* x = gsl_multifit_nlinear_position(w);
    double rcond;

    gsl_multifit_nlinear_rcond(&rcond, w);

    fprintf(stderr, "iter %2zu: nugget = %.4f, sill = %.4f, range = %.4f, cond(J) = %8.4f, |f(x)| = %.4f\n",
        iter,
        gsl_vector_get(x, 0),
        gsl_vector_get(x, 1),
        gsl_vector_get(x, 2),
        1.0 / rcond,
        gsl_blas_dnrm2(f));
}

int gaussianModel(const gsl_vector* x, void* data, gsl_vector* f)
{
    size_t n = ((struct data*)data)->n;
    double* h = ((struct data*)data)->h;
    double* y = ((struct data*)data)->y;

    double nugget = gsl_vector_get(x, 0);
    double sill = gsl_vector_get(x, 1);
    double range = gsl_vector_get(x, 2);

    size_t i;

    for (i = 0; i < n; i++)
    {
        double Yi = nugget + sill * (1.0 - exp(-((h[i] * h[i]) / (range * range))));
        gsl_vector_set(f, i, Yi - y[i]);
    }

    return GSL_SUCCESS;
}

TheoreticalParam fitTheoreticalFunction(EmpiricalVariogram* variogram) {
    const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace* w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params =
        gsl_multifit_nlinear_default_parameters();
    const size_t n = variogram->distances.size();
    const size_t p = 3;

    gsl_vector* f;
    gsl_matrix* J;
    gsl_matrix* covar = gsl_matrix_alloc(p, p);
    double* h = &variogram->distances[0];
    double* y = &variogram->values[0];
    struct data d = { n, h, y };
    double x_init[3] = { 0, 1, variogram->distances[(variogram->distances.size() - 1) / 3] };
    gsl_vector_view x = gsl_vector_view_array(x_init, p);
    gsl_rng* r;
    double chisq, chisq0;
    int status, info;
    size_t i;

    const double xtol = 1e-8;
    const double gtol = 1e-8;
    const double ftol = 0.0;

    gsl_rng_env_setup();
    r = gsl_rng_alloc(gsl_rng_default);
    fdf.f = gaussianModel;
    fdf.df = NULL;
    fdf.fvv = NULL;
    fdf.n = n;
    fdf.p = p;
    fdf.params = &d;
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

    gsl_multifit_nlinear_init(&x.vector, &fdf, w);

    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);

    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, callback, NULL, &info, w);

    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    gsl_blas_ddot(f, f, &chisq);

    #define FIT(i) gsl_vector_get(w->x, i)
    #define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

    fprintf(stderr, "summary from method '%s/%s'\n", gsl_multifit_nlinear_name(w), gsl_multifit_nlinear_trs_name(w));
    fprintf(stderr, "number of iterations: %zu\n", gsl_multifit_nlinear_niter(w));
    fprintf(stderr, "function evaluations: %zu\n", fdf.nevalf);
    fprintf(stderr, "Jacobian evaluations: %zu\n", fdf.nevaldf);
    fprintf(stderr, "reason for stopping: %s\n", (info == 1) ? "small step size" : "small gradient");
    fprintf(stderr, "initial |f(x)| = %f\n", sqrt(chisq0));
    fprintf(stderr, "final   |f(x)| = %f\n", sqrt(chisq));

    {
        double dof = n - p;
        double c = GSL_MAX_DBL(1, sqrt(chisq / dof));

        fprintf(stderr, "chisq/dof = %g\n", chisq / dof);

        fprintf(stderr, "nugget      = %.5f +/- %.5f\n", FIT(0), c * ERR(0));
        fprintf(stderr, "sill = %.5f +/- %.5f\n", FIT(1), c * ERR(1));
        fprintf(stderr, "range   = %.5f +/- %.5f\n", FIT(2), c * ERR(2));
    }

    fprintf(stderr, "status = %s\n", gsl_strerror(status));
    TheoreticalParam param{ FIT(0),FIT(1), FIT(2) };
    gsl_multifit_nlinear_free(w);
    gsl_matrix_free(covar);
    gsl_rng_free(r);

    return param;
}