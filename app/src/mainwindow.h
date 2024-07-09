#include <renderer.h>

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QWidget>

#include <QVTKOpenGLNativeWidget.h>

#include <common-includes/vtk.h>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

    std::string getTiffPath() const;
    std::string getJsonPath() const;

private slots:
    void onTiffBrowseButtonClicked();
    void onJsonBrowseButtonClicked();
    void onRenderButtonClicked();

private:
    QLineEdit* tiffPathField;
    QLineEdit* jsonPathField;

    QVTKOpenGLNativeWidget* vtkWidget;
    vtkSmartPointer<vtkRenderer> renderer;
    
    Renderer* meshRenderer;
};
