#include "mainwindow.h"

#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>
#include <logging.h>
#include <client.h>
#include <plotting.h>

#include <iostream>
#include <string>

#include <QApplication>
#include <QDir>

#ifdef _WIN32
#include <windows.h>
#endif

QString getApplicationDirPath()
{
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(NULL, buffer, MAX_PATH);
    QString path = QString::fromWCharArray(buffer);
    path = QDir::toNativeSeparators(path);
    return QFileInfo(path).absolutePath();
#else
    // Linux maybe?
    return QString();
#endif
}

int main(int argc, char* argv[])
{
    LayerBuilder layerBuilder("pecs","./res/raster.tif","./res/boreholes.json");
    layerBuilder.buildLayers();
//     Logger::init("../logs/app.log");

//     QString appDirPath = getApplicationDirPath();
//     QString pluginPath;

// #ifdef _DEBUG
//     pluginPath = QDir(appDirPath).absoluteFilePath("../../vcpkg_installed/x64-windows/debug/Qt6/plugins");
// #else
//     pluginPath = QDir(appDirPath).absoluteFilePath("../../vcpkg_installed/x64-windows/Qt6/plugins");
// #endif
//     QCoreApplication::setLibraryPaths(QStringList() << pluginPath);

//     QApplication app(argc, argv);

//     MainWindow window;
//     window.show();

//     return app.exec();
}
