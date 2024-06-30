#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>

#include "mainwindow.h"
#include <QApplication>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    // Get input
    QApplication app(argc, argv);
    MainWindow window;
    window.show();
    app.exec();

    const std::string region = window.getRegion();
    const std::string observationDataPath = window.getJsonPath();
    const std::string tiffPath = window.getTiffPath();

    // Show output
    // LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

    // ModellerSet modeller(layerBuilder);
    // modeller.createMeshes();

    // Renderer renderer{};
    // renderer.addMeshes(modeller.getMeshes());

    // // Describe what you want to be rendered
    // renderer.prepareEdges();
    // renderer.prepareSurfaces();
    // renderer.prepareLayerBody();

    // // Render
    // renderer.render();
}
