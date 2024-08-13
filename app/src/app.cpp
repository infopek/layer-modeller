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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[])
{
    Logger::init("../../../logs/app.log");

    QCoreApplication::addLibraryPath("../../vcpkg_installed/x64-windows/debug/Qt6/plugins");

    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}
