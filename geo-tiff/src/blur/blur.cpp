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
    // cv::Mat srcMat(height, width, CV_32FC1, (void*)src);
    // cv::Mat dstMat(height, width, CV_32FC1, (void*)dst);

    // cv::medianBlur(srcMat, dstMat, kernelRadius);
}

