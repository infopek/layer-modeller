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

    inline const std::string getTiffPath() const { return m_tiffPathField->text().toStdString(); }
    inline const std::string getJsonPath() const { return m_jsonPathField->text().toStdString(); }
    inline const std::string getRegion() const { return m_regionField->text().toStdString(); }

private slots:
    void onTiffBrowseButtonClicked();
    void onJsonBrowseButtonClicked();
    void onRenderButtonClicked();

private:
    QLineEdit* m_tiffPathField;
    QLineEdit* m_jsonPathField;
    QLineEdit* m_regionField;

    QVTKOpenGLNativeWidget* m_vtkWidget;
    vtkSmartPointer<vtkRenderer> m_renderer;

    Renderer* m_meshRenderer;
};
