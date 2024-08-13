#pragma once

#include <vector>

class Blur
{
public:
	Blur();
	~Blur();

	static void gaussFilter(const float* src, float* dst, int width, int height, int kernelRadius, float sigma);

	static void medianFilter(const float* src, float* dst, int width, int height, int kernelRadius);

private:
	static std::vector<float> generateGaussianKernel(int kernelRadius, float sigma);

	static void gaussFilterCUDA(const float* src, float* dst, int width, int height, int kernelRadius, float sigma);

	static void medianFilterCUDA(const float* src, float* dst, int width, int height, int kernelRadius);
};
