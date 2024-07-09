#include "mainwindow.h"

#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>

#include <iostream> // for std::cout

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), meshRenderer(nullptr)
{
    // Create the main widget
    QWidget* mainWidget = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout(mainWidget);

    // Create the TIFF and JSON path input fields with browse buttons
    QHBoxLayout* tiffLayout = new QHBoxLayout;
    tiffPathField = new QLineEdit(this);
    QPushButton* tiffBrowseButton = new QPushButton("Browse TIFF", this);
    tiffLayout->addWidget(tiffPathField);
    tiffLayout->addWidget(tiffBrowseButton);

    QHBoxLayout* jsonLayout = new QHBoxLayout;
    jsonPathField = new QLineEdit(this);
    QPushButton* jsonBrowseButton = new QPushButton("Browse JSON", this);
    jsonLayout->addWidget(jsonPathField);
    jsonLayout->addWidget(jsonBrowseButton);

    // Add these layouts to the main layout
    layout->addLayout(tiffLayout);
    layout->addLayout(jsonLayout);

    // Create the custom VTK widget
    vtkWidget = new QVTKOpenGLNativeWidget(this);
    layout->addWidget(vtkWidget);

    // Create the Render button
    QPushButton* renderButton = new QPushButton("Render", this);
    layout->addWidget(renderButton);

    setCentralWidget(mainWidget);

    // Setup VTK
    renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkWidget->renderWindow()->AddRenderer(renderer);
    meshRenderer = new Renderer(renderer);

    // Connect signals and slots
    connect(tiffBrowseButton, &QPushButton::clicked, this, &MainWindow::onTiffBrowseButtonClicked);
    connect(jsonBrowseButton, &QPushButton::clicked, this, &MainWindow::onJsonBrowseButtonClicked);
    connect(renderButton, &QPushButton::clicked, this, &MainWindow::onRenderButtonClicked);
}

MainWindow::~MainWindow()
{
    delete meshRenderer;
}

std::string MainWindow::getTiffPath() const
{
    return tiffPathField->text().toStdString();
}

std::string MainWindow::getJsonPath() const
{
    return jsonPathField->text().toStdString();
}

void MainWindow::onTiffBrowseButtonClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open TIFF File", "", "TIFF Files (*.tiff *.tif)");
    tiffPathField->setText(filePath);
}

void MainWindow::onJsonBrowseButtonClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open JSON File", "", "JSON Files (*.json)");
    jsonPathField->setText(filePath);
}

void MainWindow::onRenderButtonClicked()
{
    const std::string region = "sidjfisjd";
    const std::string observationDataPath = getJsonPath();
    const std::string tiffPath = getTiffPath();

    LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

    ModellerSet modeller(layerBuilder);
    modeller.createMeshes();

    meshRenderer->addMeshes(modeller.getMeshes());

    // Describe what you want to be rendered
    // meshRenderer->prepareEdges();
    meshRenderer->prepareSurfaces();
    // meshRenderer->prepareLayerBody();

    renderer->ResetCamera();
    vtkWidget->renderWindow()->Render();
}