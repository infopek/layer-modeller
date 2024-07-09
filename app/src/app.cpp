#include "mainwindow.h"

#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>

#include <iostream>
#include <string>

#include <QApplication>

int main(int argc, char* argv[])
{
    QCoreApplication::addLibraryPath("../../vcpkg_installed/x64-windows/debug/Qt6/plugins");

    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}

// int main(int argc, char* argv[])
// {
//     // Get input
//     QApplication app(argc, argv);
//     MainWindow window;
//     window.show();
//     app.exec();

//     const std::string region = window.getRegion();
//     
// }
