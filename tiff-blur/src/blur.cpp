#include <blur/blur.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

Blur::Blur()
{
}

Blur::~Blur()
{
}

void Blur::boxFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize)
{
    cv::Mat image = cv::imread(intputFilepath);
    cv::boxFilter(image, image, -1, cv::Size(kernelSize, kernelSize));
    std::cout << cv::imwrite(outputFilepath, image);
}

void Blur::gaussFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize, float sigma)
{
    cv::Mat image = cv::imread(intputFilepath);
    cv::GaussianBlur(image, image, cv::Size(kernelSize, kernelSize), sigma, sigma);
    std::cout << cv::imwrite(outputFilepath, image);
}

void Blur::medianFilter(const std::string& intputFilepath, const std::string& outputFilepath, int kernelSize)
{
    cv::Mat image = cv::imread(intputFilepath);
    cv::medianBlur(image, image, kernelSize);
    cv::imwrite(outputFilepath, image);
}

float Blur::gaussFunction(float x, float y, float sigma)
{
    float halfInvSigmaSqr = 0.5f / (sigma * sigma);

    float denom = 0.3183098f * halfInvSigmaSqr;	// 1 / (2 * pi * sigma ^ 2)
    float exponent = -(x * x + y * y) * (halfInvSigmaSqr);	// - (x ^ 2 + y ^ 2) / (2 * sigma ^ 2)
    return denom * expf(exponent);	// 1 / (2 * pi * sigma^2) * e ^ [-(x ^ 2 + y ^ 2) / (2 * sigma ^ 2)]
}

void Blur::calculateGaussKernel(float* gaussKernel, int kernelRadius, float sigma)
{
    int diameter = 2 * kernelRadius + 1;

    // Fill kernel
    float sum = 0.0f;
    for (int y = -kernelRadius; y <= kernelRadius; ++y)
    {
        for (int x = -kernelRadius; x <= kernelRadius; ++x)
        {
            int weightIdx = (y + kernelRadius) * diameter + (x + kernelRadius);
            gaussKernel[weightIdx] = gaussFunction(static_cast<float>(x), static_cast<float>(y), sigma);
            sum += gaussKernel[weightIdx];
        }
    }

    // Normalize kernel
    std::for_each(gaussKernel, gaussKernel + diameter * diameter,
        [=](float& f) {
            f /= sum;
        });
}