#pragma once

#include <QMainWindow>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QFormLayout>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    void browseJson();
    void browseTiff();

private:
    QLineEdit* regionLineEdit;
    QLineEdit* jsonPathLineEdit;
    QLineEdit* tiffPathLineEdit;
    QPushButton* jsonBrowseButton;
    QPushButton* tiffBrowseButton;
};
