#pragma once

#include <string>

class Blur
{
public:
	Blur();
	~Blur();

	void boxFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize);

	void gaussFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize, float sigma);

	void medianFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize);

private:
	/// @brief Calculates the Gaussian function value at (x, y) with sigma_x = sigma_y = sigma.
	/// @returns 1 / (2 * pi * sigma ^ 2) * exp(-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)).
	static float gaussFunction(float x, float y, float sigma);

	/// @brief Fills gaussKernel with values calculated via Gaussian function.
	static void calculateGaussKernel(float* gaussKernel, int kernelSize, float sigma);
};
