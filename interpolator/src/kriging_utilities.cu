#include "kriging_utilities.cuh"

double gaussianFunction(double nugget, double sill, double range, double h) {
    return nugget + sill * (1.0 - exp(-((h * h) / (range * range))));
}
