// #pragma once

// #include <QMainWindow>
// #include <QLineEdit>
// #include <QPushButton>
// #include <QFileDialog>
// #include <QFormLayout>
// #include <QVBoxLayout>
// #include <QVTKApplication.h>

// #include <vtkRenderer.h>


// class MainWindow : public QMainWindow
// {
// public:
//     MainWindow(QWidget* parent = nullptr);
//     ~MainWindow();

//     inline std::string getRegion() const { return regionLineEdit->text().toStdString(); }

//     inline std::string getJsonPath() const { return jsonPathLineEdit->text().toStdString(); }

//     inline std::string getTiffPath() const { return tiffPathLineEdit->text().toStdString(); }

// private:
//     void browseJson();
//     void browseTiff();

//     void closeWindow();

// private:
//     QLineEdit* regionLineEdit;
//     QLineEdit* jsonPathLineEdit;
//     QLineEdit* tiffPathLineEdit;
//     QPushButton* jsonBrowseButton;
//     QPushButton* tiffBrowseButton;
//     QPushButton* okButton;
// };
