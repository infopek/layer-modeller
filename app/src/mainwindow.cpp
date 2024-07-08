// #include "mainwindow.h"

// MainWindow::MainWindow(QWidget* parent)
//     : QMainWindow(parent)
// {
//     // Set the size of the window
//     resize(1200, 800); // Adjust as necessary

//     QWidget* centralWidget = new QWidget(this);
//     setCentralWidget(centralWidget);

//     QFormLayout* formLayout = new QFormLayout;

//     regionLineEdit = new QLineEdit();
//     formLayout->addRow("Region:", regionLineEdit);

//     jsonPathLineEdit = new QLineEdit();
//     jsonPathLineEdit->setReadOnly(true); // Make it read-only
//     formLayout->addRow("JSON File:", jsonPathLineEdit);

//     jsonBrowseButton = new QPushButton("Browse");
//     connect(jsonBrowseButton, &QPushButton::clicked, this, &MainWindow::browseJson);
//     formLayout->addRow(jsonBrowseButton);

//     tiffPathLineEdit = new QLineEdit();
//     tiffPathLineEdit->setReadOnly(true); // Make it read-only
//     formLayout->addRow("TIFF File:", tiffPathLineEdit);

//     tiffBrowseButton = new QPushButton("Browse");
//     connect(tiffBrowseButton, &QPushButton::clicked, this, &MainWindow::browseTiff);
//     formLayout->addRow(tiffBrowseButton);

//     QVBoxLayout* leftLayout = new QVBoxLayout;
//     leftLayout->addLayout(formLayout);

//     // Create a VTK widget
//     vtkWidget = new QVTKWidget;
//     leftLayout->addWidget(vtkWidget);

//     centralWidget->setLayout(leftLayout);

//     // Initialize VTK renderer and render window
//     renderer = vtkRenderer::New();
//     vtkWidget->GetRenderWindow()->AddRenderer(renderer);
//     renderer->SetBackground(0.8, 0.8, 0.8); // Set renderer background color

//     // Set up any initial VTK scene or interaction here

//     okButton = new QPushButton("OK");
//     connect(okButton, &QPushButton::clicked, this, &MainWindow::closeWindow);

//     leftLayout->addWidget(okButton);
// }

// MainWindow::~MainWindow()
// {
// }

// void MainWindow::browseJson() {
//     QString fileName = QFileDialog::getOpenFileName(this, "Open JSON File", "", "JSON Files (*.json)");
//     if (!fileName.isEmpty()) {
//         jsonPathLineEdit->setText(fileName);
//     }
// }

// void MainWindow::browseTiff() {
//     QString fileName = QFileDialog::getOpenFileName(this, "Open TIFF File", "", "TIFF Files (*.tif *.tiff)");
//     if (!fileName.isEmpty()) {
//         tiffPathLineEdit->setText(fileName);
//     }
// }

// void MainWindow::closeWindow()
// {
//     close();
// }
