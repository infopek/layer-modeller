#include <blur/blur.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_utils.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>

#include <algorithm>
#include <cmath>
#include <numbers>

__device__ unsigned char* dev_src;
__device__ unsigned char* dev_dst;
__device__ float* dev_weights;

extern __shared__ unsigned char sh_data[];

__global__ static void blurImageGauss(const unsigned char* src, unsigned char* dst, float* weights, size_t width, size_t height, int tileSize, int kernelRadius, float sigma)
{
	int diameter = 2 * kernelRadius + 1;

	// Shared data
	unsigned char* sharedImageData = sh_data;
	float* sharedWeights = (float*)&sharedImageData[blockDim.x * blockDim.y];

	// Calculate CUDA indices
	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fminf(fmaxf(0, x), width - 1);
	y = fminf(fmaxf(0, y), height - 1);

	// Calculate indices
	unsigned int idx = y * width + x;
	unsigned int blockIdx = threadIdx.y * blockDim.x + threadIdx.x;
	unsigned int weightIdx = threadIdx.y * diameter + threadIdx.x;

	// Each thread in a block copies a pixel to shared block from src image
	sharedImageData[blockIdx] = src[idx];
	__syncthreads();

	// Some threads also copy weights
	if (weightIdx < diameter * diameter)
		sharedWeights[weightIdx] = weights[weightIdx];

	// Apply gauss blur
	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		float sum = 0.0f;
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				sum += (float)sharedImageData[blockIdx + (r * blockDim.x) + c] * sharedWeights[(r + kernelRadius) * diameter + (c + kernelRadius)];

		dst[idx] = (unsigned char)sum;
	}
}

__global__ static void blurImageBox(const unsigned char* src, unsigned char* dst, size_t width, size_t height, int kernelRadius, int tileSize)
{
	// Get current position of thread in image
	int x = blockIdx.x * tileSize + threadIdx.x - kernelRadius;
	int y = blockIdx.y * tileSize + threadIdx.y - kernelRadius;

	// Clamp position to edge of image
	x = fminf(fmaxf(0, x), width - 1);
	y = fminf(fmaxf(0, y), height - 1);

	// Calculate global and (block) local index
	unsigned int idx = y * width + x;
	unsigned int blockIdx = blockDim.x * threadIdx.y + threadIdx.x;

	// Each thread in a block copies a pixel from src to shared memory
	sh_data[blockIdx] = src[idx];
	__syncthreads();

	if (threadIdx.x >= kernelRadius && threadIdx.y >= kernelRadius && threadIdx.x < (blockDim.x - kernelRadius) && threadIdx.y < (blockDim.y - kernelRadius))
	{
		// Use average of kernel to apply blur effect
		float sum = 0.0f;
		for (int r = -kernelRadius; r <= kernelRadius; ++r)
			for (int c = -kernelRadius; c <= kernelRadius; ++c)
				sum += (float)sh_data[blockIdx + (r * blockDim.x) + c];

		int diameter = 2 * kernelRadius + 1;
		dst[idx] = (unsigned char)(sum / (diameter * diameter));
	}
}

constexpr int c_maskRadius = 3;
constexpr int c_maskSize = c_maskRadius * c_maskRadius;

__global__ static void blurImageMedian(unsigned char* src, unsigned char* dst, size_t width, size_t height)
{
	const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

	// Fill mask with corresponding image pixels
	sh_data[threadIdx.y * blockDim.x + threadIdx.x] = src[y * width + x];
	__syncthreads();

	// The first thread sorts the mask using insertion sort
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int i = 1;
		while (i < c_maskSize)
		{
			int x = sh_data[i];
			int j = i - 1;
			while (j >= 0 && sh_data[j] > x)
			{
				sh_data[j + 1] = sh_data[j];
				--j;
			}
			sh_data[j + 1] = x;
			++i;
		}
	}
	__syncthreads();

	dst[y * width + x] = sh_data[c_maskSize / 2];	// maskSize / 2 => median of mask
}

// const unsigned char* src;
	// unsigned char* dst;

	// // Init CUDA memory
	// size_t imageSize = width * height * sizeof(unsigned char);

	// CUDA_CALL(cudaMalloc(&dev_src, imageSize));
	// CUDA_CALL(cudaMalloc(&dev_dst, imageSize));

	// // Copy source buffer to device buffer
	// CUDA_CALL(cudaMemcpy(dev_src, src, imageSize, cudaMemcpyHostToDevice));

	// // Kernel launch dimensions & parameters
	// dim3 blockSize(16, 16);
	// int tileSize = blockSize.x - 2 * kernelRadius;

	// dim3 gridSize(width / tileSize + 1,
	// 	height / tileSize + 1);

	// size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char);

	// // Launch kernel
	// blurImageBox KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, width, height, kernelRadius, tileSize);
	// CUDA_CALL(cudaDeviceSynchronize());

	// // Copy output to dst
	// CUDA_CALL(cudaMemcpy(dst, dev_dst, imageSize, cudaMemcpyDeviceToHost));

	// // Cleanup
	// CUDA_CALL(cudaFree(dev_src));
	// CUDA_CALL(cudaFree(dev_dst));


// const unsigned char* src;
	// unsigned char* dst;

	// // Fill Gauss kernel
	// int diameter = 2 * kernelRadius + 1;
	// float* weights = new float[diameter * diameter];
	// calculateGaussKernel(weights, kernelRadius, sigma);

	// // Init CUDA memory
	// size_t imageSize = width * height * sizeof(unsigned char);
	// size_t kernelSize = diameter * diameter * sizeof(float);

	// CUDA_CALL(cudaMalloc(&dev_src, imageSize));
	// CUDA_CALL(cudaMalloc(&dev_dst, imageSize));
	// CUDA_CALL(cudaMalloc(&dev_weights, kernelSize));

	// CUDA_CALL(cudaMemcpy(dev_src, src, imageSize, cudaMemcpyHostToDevice));
	// CUDA_CALL(cudaMemcpy(dev_weights, weights, kernelSize, cudaMemcpyHostToDevice));

	// // Kernel launch dimensions & parameters
	// dim3 blockSize(16, 16);
	// int tileSize = blockSize.x - 2 * kernelRadius;

	// dim3 gridSize(width / tileSize + 1,
	// 	height / tileSize + 1);

	// size_t sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char) + kernelSize;

	// // Launch kernel
	// blurImageGauss KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, dev_weights, width, height, tileSize, kernelRadius, sigma);
	// CUDA_CALL(cudaDeviceSynchronize());

	// // Copy output to dst
	// CUDA_CALL(cudaMemcpy(dst, dev_dst, imageSize, cudaMemcpyDeviceToHost));

	// // Cleanup
	// CUDA_CALL(cudaFree(dev_src));
	// CUDA_CALL(cudaFree(dev_dst));
	// CUDA_CALL(cudaFree(dev_weights));
	// delete[] weights;

// const unsigned char* src;
	// unsigned char* dst;

	// size_t size = width * height;
	// CUDA_CALL(cudaMalloc<unsigned char>(&dev_src, size));
	// CUDA_CALL(cudaMalloc<unsigned char>(&dev_dst, size));

	// CUDA_CALL(cudaMemcpy(dev_src, src, size, cudaMemcpyHostToDevice));

	// dim3 blockSize(c_maskRadius, c_maskRadius);
	// dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	// size_t sharedMemSize = c_maskSize * sizeof(unsigned char);
	// blurImageMedian KERNEL_ARGS3(gridSize, blockSize, sharedMemSize)(dev_src, dev_dst, width, height);

	// CUDA_CALL(cudaDeviceSynchronize());

	// CUDA_CALL(cudaMemcpy(dst, dev_dst, size, cudaMemcpyDeviceToHost));

	// CUDA_CALL(cudaFree(dev_src));
	// CUDA_CALL(cudaFree(dev_dst));