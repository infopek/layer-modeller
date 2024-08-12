#include <blur/blur.h>


#include "../cuda_utils.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gaussBlurHorizontal(const float* src, float* dst, int width, int height, const float* kernel, int kernelRadius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	float sum = 0.0f;
	for (int k = -kernelRadius; k <= kernelRadius; ++k)
	{
		int xSample = fminf(fmaxf(x + k, 0), width - 1);
		sum += src[y * width + xSample] * kernel[kernelRadius + k];
	}

	dst[y * width + x] = sum;
}

__global__ void gaussBlurVertical(const float* src, float* dst, int width, int height, const float* kernel, int kernelRadius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	float sum = 0.0f;
	for (int k = -kernelRadius; k <= kernelRadius; ++k)
	{
		int ySample = fminf(fmaxf(y + k, 0), height - 1);
		sum += src[ySample * width + x] * kernel[kernelRadius + k];
	}

	dst[y * width + x] = sum;
}

std::vector<float> Blur::generateGaussianKernel(int kernelRadius, float sigma)
{
	int diameter = 2 * kernelRadius + 1;
	std::vector<float> kernel(diameter);
	float sum = 0.0f;

	for (int i = -kernelRadius; i <= kernelRadius; ++i)
	{
		kernel[i + kernelRadius] = expf(-(i * i) / (2.0f * sigma * sigma));
		sum += kernel[i + kernelRadius];
	}

	// Normalize the kernel
	for (int i = 0; i < diameter; ++i)
		kernel[i] /= sum;

	return kernel;
}

void Blur::gaussFilterCUDA(const float* src, float* dst, int width, int height, int kernelRadius, float sigma)
{
	// Generate Gaussian Kernel
	std::vector<float> h_kernel = generateGaussianKernel(kernelRadius, sigma);

	// Allocate device memory
	float* d_src;
	float* d_dst;
	float* d_intermediate;
	float* d_kernel;

	size_t imageSize = width * height * sizeof(float);
	size_t kernelSize = (2 * kernelRadius + 1) * sizeof(float);

	CUDA_CALL(cudaMalloc(&d_src, imageSize));
	CUDA_CALL(cudaMalloc(&d_dst, imageSize));
	CUDA_CALL(cudaMalloc(&d_intermediate, imageSize));
	CUDA_CALL(cudaMalloc(&d_kernel, kernelSize));

	CUDA_CALL(cudaMemcpy(d_src, src, imageSize, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_kernel, h_kernel.data(), kernelSize, cudaMemcpyHostToDevice));

	// Kernel launch dimensions
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	// Horizontal pass
	gaussBlurHorizontal KERNEL_ARGS2(gridSize, blockSize)(d_src, d_intermediate, width, height, d_kernel, kernelRadius);
	cudaDeviceSynchronize();

	// Vertical pass
	gaussBlurVertical KERNEL_ARGS2(gridSize, blockSize)(d_intermediate, d_dst, width, height, d_kernel, kernelRadius);
	cudaDeviceSynchronize();

	// Copy result back to host
	CUDA_CALL(cudaMemcpy(dst, d_dst, imageSize, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA_CALL(cudaFree(d_src));
	CUDA_CALL(cudaFree(d_dst));
	CUDA_CALL(cudaFree(d_intermediate));
	CUDA_CALL(cudaFree(d_kernel));
}

void Blur::medianFilterCUDA(const float* src, float* dst, int width, int height)
{
	// size_t size = width * height;
	// CUDA_CALL(cudaMalloc<float>(&dev_src, size));
	// CUDA_CALL(cudaMalloc<float>(&dev_dst, size));

	// CUDA_CALL(cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice));

	// dim3 blockSize(c_maskRadius, c_maskRadius);
	// dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	// size_t sharedMemSize = c_maskSize * sizeof(float);
	// blurImageMedian KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, width, height);

	// CUDA_CALL(cudaDeviceSynchronize());

	// CUDA_CALL(cudaMemcpy(dst, dev_dst, size, cudaMemcpyDeviceToHost));

	// CUDA_CALL(cudaFree(dev_src));
	// CUDA_CALL(cudaFree(dev_dst));
}
