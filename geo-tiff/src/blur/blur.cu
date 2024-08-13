#include <blur/blur.h>
#include "../cuda_utils.h"

#ifndef __CUDACC__
#define __syncthreads()
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <cmath>

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

__global__ void medianBlur(const float* src, float* dst, int width, int height, int kernelRadius)
{
	extern __shared__ float sh_data[];

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int sharedWidth = blockDim.x + 2 * kernelRadius;
	int sharedHeight = blockDim.y + 2 * kernelRadius;

	int x_shared = threadIdx.x + kernelRadius;
	int y_shared = threadIdx.y + kernelRadius;

	// Load data into shared memory with boundary checks
	for (int i = threadIdx.y; i < sharedHeight; i += blockDim.y)
	{
		for (int j = threadIdx.x; j < sharedWidth; j += blockDim.x)
		{
			int x_global = (int)fminf(fmaxf(blockIdx.x * blockDim.x + j - kernelRadius, 0), width - 1);
			int y_global = (int)fminf(fmaxf(blockIdx.y * blockDim.y + i - kernelRadius, 0), height - 1);
			sh_data[i * sharedWidth + j] = src[y_global * width + x_global];
		}
	}

	__syncthreads();

	// Only work on valid pixels within the original block
	if (x < width && y < height)
	{
		float window[225]; // Max kernel size 15x15
		int index = 0;

		for (int i = -kernelRadius; i <= kernelRadius; ++i)
		{
			for (int j = -kernelRadius; j <= kernelRadius; ++j)
			{
				window[index++] = sh_data[(y_shared + i) * sharedWidth + (x_shared + j)];
			}
		}

		// Sort the window array to find the median
		for (int i = 0; i < index - 1; ++i)
		{
			for (int j = i + 1; j < index; ++j)
			{
				if (window[i] > window[j])
				{
					float temp = window[i];
					window[i] = window[j];
					window[j] = temp;
				}
			}
		}

		// Set the median value in the output array
		dst[y * width + x] = window[index / 2];
	}
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
	CUDA_CALL(cudaDeviceSynchronize());

	// Vertical pass
	gaussBlurVertical KERNEL_ARGS2(gridSize, blockSize)(d_intermediate, d_dst, width, height, d_kernel, kernelRadius);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy result back to host
	CUDA_CALL(cudaMemcpy(dst, d_dst, imageSize, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA_CALL(cudaFree(d_src));
	CUDA_CALL(cudaFree(d_dst));
	CUDA_CALL(cudaFree(d_intermediate));
	CUDA_CALL(cudaFree(d_kernel));
}

void Blur::medianFilterCUDA(const float* src, float* dst, int width, int height, int kernelRadius)
{
	// Allocate device memory
	float* d_src;
	float* d_dst;
	size_t imageSize = width * height * sizeof(float);

	CUDA_CALL(cudaMalloc(&d_src, imageSize));
	CUDA_CALL(cudaMalloc(&d_dst, imageSize));

	CUDA_CALL(cudaMemcpy(d_src, src, imageSize, cudaMemcpyHostToDevice));

	// Kernel launch dimensions
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	size_t sharedMemSize = (blockSize.x + 2 * kernelRadius) * (blockSize.y + 2 * kernelRadius) * sizeof(float);

	// Launch median filter kernel
	medianBlur KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(d_src, d_dst, width, height, kernelRadius);
	CUDA_CALL(cudaDeviceSynchronize());

	// Copy result back to host
	CUDA_CALL(cudaMemcpy(dst, d_dst, imageSize, cudaMemcpyDeviceToHost));

	// Free device memory
	CUDA_CALL(cudaFree(d_src));
	CUDA_CALL(cudaFree(d_dst));
}
