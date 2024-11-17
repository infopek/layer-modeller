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

    // Create generate button
    m_generateButton = new QPushButton("Generate", this);
    layout->addWidget(m_generateButton);

    // Create the Render button
    m_renderButton = new QPushButton("Render", this);
    layout->addWidget(m_renderButton);

    // Export to shapefile / vtk file
    m_exportButton = new QPushButton("Export", this);
    layout->addWidget(m_exportButton);

    setCentralWidget(mainWidget);

    // Setup VTK and Renderer
    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_vtkWidget->renderWindow()->AddRenderer(m_renderer);
    m_meshRenderer = new Renderer(m_renderer);
    m_shapefileGenerator = new ShapefileGenerator();

    // Connect signals and slots
    connect(tiffBrowseButton, &QPushButton::clicked, this, &MainWindow::onTiffBrowseButtonClicked);
    connect(jsonBrowseButton, &QPushButton::clicked, this, &MainWindow::onJsonBrowseButtonClicked);
    connect(m_regionField, &QLineEdit::textChanged, this, &MainWindow::onRegionFieldTextChanged);

    connect(m_generateButton, &QPushButton::clicked, this, &MainWindow::onGenerateButtonClicked);
    connect(m_renderButton, &QPushButton::clicked, this, &MainWindow::onRenderButtonClicked);
    connect(m_exportButton, &QPushButton::clicked, this, &MainWindow::onExportButtonClicked);

    setButtonStates(false, false, false);
}

MainWindow::~MainWindow()
{
    delete m_meshRenderer;
    delete m_shapefileGenerator;
}

void MainWindow::onTiffBrowseButtonClicked()
{
    setButtonStates(false, false, false);

    QString filePath = QFileDialog::getOpenFileName(this, "Open TIFF File", "", "TIFF Files (*.tiff *.tif)");
    m_tiffPathField->setText(filePath);

    if (!filePath.isEmpty() && !m_jsonPathField->text().isEmpty())
        setButtonStates(true, false, false);
}

void MainWindow::onJsonBrowseButtonClicked()
{
    QString filePath = QFileDialog::getOpenFileName(this, "Open JSON File", "", "JSON Files (*.json)");
    m_jsonPathField->setText(filePath);

    if (!filePath.isEmpty() && !m_tiffPathField->text().isEmpty())
        setButtonStates(true, false, false);
}

void MainWindow::onRegionFieldTextChanged(const QString& text)
{
    if (!text.isEmpty())
        setButtonStates(true, false, false);
}

void MainWindow::onGenerateButtonClicked()
{
    const std::string region = getRegion();
    const std::string observationDataPath = getJsonPath();
    const std::string tiffPath = getTiffPath();

    setButtonStates(false, false, false);

    // Build layers
    auto _ = QtConcurrent::run(
        [this, region, observationDataPath, tiffPath]()
        {
            try {
                LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

                ModellerSet modeller(layerBuilder);
                modeller.createMeshes();

                m_generatedMeshes = modeller.getMeshes();

                // Invoke the method on the main thread
                QMetaObject::invokeMethod(this, "onGeneratingComplete", Qt::QueuedConnection);
            }
            catch (const std::exception& e) {
                std::cerr << "Exception caught: " << e.what() << std::endl;
            }
        });
}

void MainWindow::onRenderButtonClicked()
{
    // Setup
    m_meshRenderer->clear();
    m_renderer->RemoveAllViewProps();
    m_renderer->ResetCamera();
    m_vtkWidget->renderWindow()->Render();

    setButtonStates(false, false, false);

    // Add meshes
    m_meshRenderer->addMeshes(m_generatedMeshes);
    m_meshRenderer->prepareMeshes();

    // Render
    m_renderer->ResetCamera();
    m_vtkWidget->renderWindow()->Render();

    setButtonStates(true, true, true);
}

void MainWindow::onExportButtonClicked()
{
    QString directoryPath = QFileDialog::getExistingDirectory(this, "Select Output Folder", "");

    if (!directoryPath.isEmpty()) {
        setButtonStates(false, false, false);

        try {
            m_shapefileGenerator->addMeshes(m_generatedMeshes);
            m_shapefileGenerator->writeVTKFile(directoryPath.toStdString());
        }
        catch (const std::exception& e) {
            std::cerr << "Exception caught: " << e.what() << std::endl;
        }

        setButtonStates(true, true, true);
    }
    else {
        std::cerr << "No directory selected!" << std::endl;
    }
}

void MainWindow::onGeneratingComplete()
{
    setButtonStates(true, true, true);
}

void MainWindow::setButtonStates(bool generateButtonState, bool renderButtonState, bool exportButtonState)
{
    m_generateButton->setEnabled(generateButtonState);
    m_renderButton->setEnabled(renderButtonState);
    m_exportButton->setEnabled(exportButtonState);
}


