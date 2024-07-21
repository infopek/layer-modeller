#pragma once

class Blur
{
public:
	Blur();
	~Blur();

	static void boxFilter(const float* src, float* dst, int width, int height, int kernelSize);

	static void gaussFilter(const float* src, float* dst, int width, int height, int kernelSize, float sigma);

	static void medianFilter(const float* src, float* dst, int width, int height, int kernelSize);

private:
	/// @brief Calculates the Gaussian function value at (x, y) with sigma_x = sigma_y = sigma.
	/// @returns 1 / (2 * pi * sigma ^ 2) * exp(-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)).
	static float gaussFunction(float x, float y, float sigma);

	/// @brief Fills gaussKernel with values calculated via Gaussian function.
	static void calculateGaussKernel(float* gaussKernel, int kernelSize, float sigma);
};
