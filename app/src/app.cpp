#include "mainwindow.h"

#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>
#include <logging.h>

#include <iostream>
#include <string>

#include <QApplication>

// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

int main(int argc, char* argv[])
{
    Logger::init("../../../logs/app.log");

    QCoreApplication::addLibraryPath("../../vcpkg_installed/x64-windows/debug/Qt6/plugins");

    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();

    // int width, height, channels;
    // unsigned char* image = stbi_load("../../../res/tiff/nature.jpg", &width, &height, &channels, 1); // Load as grayscale

    // if (!image) {
    //     std::cerr << "Failed to load image!" << std::endl;
    //     return -1;
    // }

    // // Allocate the float buffer
    // std::vector<float> srcBuffer(width * height);
    // std::vector<float> dstBuffer(width * height);

    // // Convert the image data to float
    // for (int i = 0; i < width * height; ++i)
    //     srcBuffer[i] = static_cast<float>(image[i]) / 255.0f;


    // stbi_image_free(image);

    // // Apply the Gaussian Blur
    // int kernelSize = 9;
    // float sigma = 2.0f;
    // Blur::gaussFilter(srcBuffer.data(), dstBuffer.data(), width, height, kernelSize, sigma);

    // // Convert back to unsigned char and save the image
    // std::vector<unsigned char> outputImage(width * height);

    // for (int i = 0; i < width * height; ++i) {
    //     outputImage[i] = static_cast<unsigned char>(std::min(1.0f, std::max(0.0f, dstBuffer[i])) * 255.0f);
    // }

    // stbi_write_jpg("../../../res/tiff/nature_blurred.jpg", width, height, 1, outputImage.data(), width);
}
