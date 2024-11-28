#include "variogram.h"

std::string VariogramCalculator::s_logPrefix = "[VARIOGRAM_CALCULATOR] --";

VariogramCalculator::VariogramCalculator(std::vector<Point> &points)
{
    auto start = std::chrono::high_resolution_clock::now();
    Logger::log(LogLevel::INFO, s_logPrefix + "Starting VariogramCalculator constructor.");

    computeEmpiricalVariogram(points);
    fitTheoreticalFunction();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Logger::log(LogLevel::INFO, s_logPrefix + "VariogramCalculator constructed in " + std::to_string(duration) + " ms.");
}

VariogramCalculator::~VariogramCalculator()
{
    Logger::log(LogLevel::INFO, s_logPrefix + "Destroying VariogramCalculator.");
}
void VariogramCalculator::callback(const size_t iter, void *params, const gsl_multifit_nlinear_workspace *w)
{
    gsl_vector* f = gsl_multifit_nlinear_residual(w);
    gsl_vector* x = gsl_multifit_nlinear_position(w);
    double rcond;

    gsl_multifit_nlinear_rcond(&rcond, w);

}

int VariogramCalculator::gaussianModel(const gsl_vector* x, void* data, gsl_vector* f)
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
void VariogramCalculator::computeEmpiricalVariogram(std::vector<Point>& points) {
    auto start = std::chrono::high_resolution_clock::now();
    Logger::log(LogLevel::INFO, s_logPrefix + "Starting computeEmpiricalVariogram with " + std::to_string(points.size()) + " points.");

    int numPoints = points.size();
    int numPairs = numPoints * (numPoints + 1) / 2;
    std::vector<int> indices(numPairs);

    for (int i = 0; i < numPairs; i++) indices[i] = i;

    double* distances = new double[numPoints * numPoints];
    double* semivariances = new double[numPoints * numPoints];

    for (size_t i = 0; i < numPoints; ++i) {
        for (size_t j = 0; j < numPoints; ++j) {
            double deltaX = static_cast<double>(points[i].x - points[j].x);
            double deltaY = static_cast<double>(points[i].y - points[j].y);
            distances[i * numPoints + j] = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            semivariances[i * numPoints + j] = 0.5 * std::pow(points[i].z - points[j].z, 2);
        }
    }
    std::sort(indices.begin(), indices.end(), [&](const int& idx1, const int& idx2) {
        return distances[idx1] < distances[idx2];
    });

    std::vector<double> semivarianceAverages(numPairs);
    std::vector<double> sortedDistances(numPairs);

    semivarianceAverages[0] = semivariances[indices[0]];
    for (int k = 1; k < numPairs; k++) {
        semivarianceAverages[k] = (semivarianceAverages[k - 1] * k + semivariances[indices[k]]) / (k + 1.0);
        sortedDistances[k] = distances[indices[k]];
    }
    variogram.empirical.values = semivarianceAverages;
    variogram.empirical.distances = sortedDistances;

    delete[] distances;
    delete[] semivariances;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Logger::log(LogLevel::INFO, s_logPrefix + "computeEmpiricalVariogram completed in " + std::to_string(duration) + " ms.");
}

void VariogramCalculator::fitTheoreticalFunction() {
    auto start = std::chrono::high_resolution_clock::now();
    Logger::log(LogLevel::INFO, s_logPrefix + "Starting fitTheoreticalFunction.");

    const gsl_multifit_nlinear_type* T = gsl_multifit_nlinear_trust;
    gsl_multifit_nlinear_workspace* w;
    gsl_multifit_nlinear_fdf fdf;
    gsl_multifit_nlinear_parameters fdf_params =
        gsl_multifit_nlinear_default_parameters();
    const size_t n = variogram.empirical.distances.size();
    const size_t p = 3;

    gsl_vector* f;
    gsl_matrix* J;
    gsl_matrix* covar = gsl_matrix_alloc(p, p);
    double* h = &variogram.empirical.distances[0];
    double* y = &variogram.empirical.values[0];
    struct data d = { n, h, y };
    double x_init[3] = { 0, 1, variogram.empirical.distances[(variogram.empirical.distances.size() - 1) / 3] };
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
    fdf.f = this->gaussianModel;
    fdf.df = NULL;
    fdf.fvv = NULL;
    fdf.n = n;
    fdf.p = p;
    fdf.params = &d;
    w = gsl_multifit_nlinear_alloc(T, &fdf_params, n, p);

    gsl_multifit_nlinear_init(&x.vector, &fdf, w);

    f = gsl_multifit_nlinear_residual(w);
    gsl_blas_ddot(f, f, &chisq0);

    status = gsl_multifit_nlinear_driver(100, xtol, gtol, ftol, this->callback, NULL, &info, w);

    J = gsl_multifit_nlinear_jac(w);
    gsl_multifit_nlinear_covar(J, 0.0, covar);

    gsl_blas_ddot(f, f, &chisq);

    #define FIT(i) gsl_vector_get(w->x, i)
    #define ERR(i) sqrt(gsl_matrix_get(covar,i,i))

    {
        double dof = n - p;
        double c = GSL_MAX_DBL(1, sqrt(chisq / dof));
        Logger::log(LogLevel::INFO, s_logPrefix + 
                    "Fitting results: Nugget = " + std::to_string(FIT(0)) +
                    ", Sill = " + std::to_string(FIT(1)) +
                    ", Range = " + std::to_string(FIT(2)));
    }
    TheoreticalParam param{ FIT(0), FIT(1), FIT(2) };
    gsl_multifit_nlinear_free(w);
    gsl_matrix_free(covar);
    gsl_rng_free(r);
    variogram.theoretical = param;

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    Logger::log(LogLevel::INFO, s_logPrefix + "fitTheoreticalFunction completed in " + std::to_string(duration) + " ms.");
}
