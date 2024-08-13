#include <blur/blur.h>

#include <iostream>

Blur::Blur()
{
}

Blur::~Blur()
{
}

void Blur::gaussFilter(const float* src, float* dst, int width, int height, int kernelRadius, float sigma)
{
    gaussFilterCUDA(src, dst, width, height, kernelRadius, sigma);
}

void Blur::medianFilter(const float* src, float* dst, int width, int height, int kernelRadius)
{
    medianFilterCUDA(src, dst, width, height, kernelRadius);
}

