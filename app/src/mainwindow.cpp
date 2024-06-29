#include "mainwindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
{
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QFormLayout* formLayout = new QFormLayout(centralWidget);

    regionLineEdit = new QLineEdit();
    formLayout->addRow("Region:", regionLineEdit);

    jsonPathLineEdit = new QLineEdit();
    formLayout->addRow("JSON File:", jsonPathLineEdit);

    jsonBrowseButton = new QPushButton("Browse");
    connect(jsonBrowseButton, &QPushButton::clicked, this, &MainWindow::browseJson);
    formLayout->addRow(jsonBrowseButton);

    tiffPathLineEdit = new QLineEdit();
    formLayout->addRow("TIFF File:", tiffPathLineEdit);

    tiffBrowseButton = new QPushButton("Browse");
    connect(tiffBrowseButton, &QPushButton::clicked, this, &MainWindow::browseTiff);
    formLayout->addRow(tiffBrowseButton);

    centralWidget->setLayout(formLayout);
}

MainWindow::~MainWindow()
{
}

void MainWindow::browseJson() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open JSON File", "", "JSON Files (*.json)");
    if (!fileName.isEmpty()) {
        jsonPathLineEdit->setText(fileName);
    }
}

void MainWindow::browseTiff() {
    QString fileName = QFileDialog::getOpenFileName(this, "Open TIFF File", "", "TIFF Files (*.tif *.tiff)");
    if (!fileName.isEmpty()) {
        tiffPathLineEdit->setText(fileName);
    }
}