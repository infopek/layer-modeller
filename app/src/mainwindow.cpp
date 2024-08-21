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
    : QMainWindow(parent), m_meshRenderer(nullptr)
{
    resize(720, 540);

    // Create the main widget
    QWidget* mainWidget = new QWidget;
    QVBoxLayout* layout = new QVBoxLayout(mainWidget);

    // Create the TIFF path input layout
    QHBoxLayout* tiffLayout = new QHBoxLayout;
    m_tiffPathField = new QLineEdit(this);
    m_tiffPathField->setReadOnly(true);
    QPushButton* tiffBrowseButton = new QPushButton("Browse TIFF", this);
    tiffLayout->addWidget(m_tiffPathField);
    tiffLayout->addWidget(tiffBrowseButton);

    // Create the JSON path input layout
    QHBoxLayout* jsonLayout = new QHBoxLayout;
    m_jsonPathField = new QLineEdit(this);
    m_jsonPathField->setReadOnly(true);
    QPushButton* jsonBrowseButton = new QPushButton("Browse JSON", this);
    jsonLayout->addWidget(m_jsonPathField);
    jsonLayout->addWidget(jsonBrowseButton);

    // Create region input field
    QHBoxLayout* regionLayout = new QHBoxLayout;
    m_regionField = new QLineEdit(this);
    m_regionField->setPlaceholderText("Enter region...");
    regionLayout->addWidget(m_regionField);

    // Add these layouts to the main layout
    layout->addLayout(tiffLayout);
    layout->addLayout(jsonLayout);
    layout->addLayout(regionLayout);

    // Create the custom VTK widget
    m_vtkWidget = new QVTKOpenGLNativeWidget(this);
    layout->addWidget(m_vtkWidget);

    // Create the Render button
    QPushButton* renderButton = new QPushButton("Render", this);
    layout->addWidget(renderButton);

    setCentralWidget(mainWidget);

    // Setup VTK and Renderer
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_vtkWidget->renderWindow()->AddRenderer(m_renderer);
    m_meshRenderer = new Renderer(m_renderer);

    // Connect signals and slots
    connect(tiffBrowseButton, &QPushButton::clicked, this, &MainWindow::onTiffBrowseButtonClicked);
    connect(jsonBrowseButton, &QPushButton::clicked, this, &MainWindow::onJsonBrowseButtonClicked);
    connect(renderButton, &QPushButton::clicked, this, &MainWindow::onRenderButtonClicked);
}

MainWindow::~MainWindow()
{
    delete m_meshRenderer;
}

void MainWindow::onTiffBrowseButtonClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open TIFF File", "", "TIFF Files (*.tiff *.tif)");
    m_tiffPathField->setText(filePath);
}

void MainWindow::onJsonBrowseButtonClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open JSON File", "", "JSON Files (*.json)");
    m_jsonPathField->setText(filePath);
}

void MainWindow::onRenderButtonClicked()
{
    const std::string region = getRegion();

    const std::string observationDataPath = getJsonPath();
    const std::string tiffPath = getTiffPath();

    QPushButton* renderButton = qobject_cast<QPushButton*>(sender());
    if (renderButton)
        renderButton->setEnabled(false);

    // Build layers
    auto _ = QtConcurrent::run(
        [this, renderButton, region, observationDataPath, tiffPath]()
        {
            LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

            ModellerSet modeller(layerBuilder);
            modeller.createMeshes();

            m_meshRenderer->addMeshes(modeller.getMeshes());

            QMetaObject::invokeMethod(this, "onGeneratingComplete", Qt::QueuedConnection, Q_ARG(QPushButton*, renderButton));
        });
}

void MainWindow::onGeneratingComplete(QPushButton* renderButton)
{
    // Describe what you want to be rendered
    m_meshRenderer->prepareEdges();
    m_meshRenderer->prepareSurfaces();
    m_meshRenderer->prepareLayerBody();

    // Render
    m_renderer->ResetCamera();
    m_vtkWidget->renderWindow()->Render();

    renderButton->setEnabled(true);
}

