#include <renderer.h>
#include <shp_generator.h>

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QFileDialog>
#include <QWidget>
#include <QThread>
#include <QtConcurrent/QtConcurrent>

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
    void onRegionFieldTextChanged(const QString& text);

    void onGenerateButtonClicked();
    void onRenderButtonClicked();
    void onExportButtonClicked();

    void onGeneratingComplete();

private:
    void setButtonStates(bool generateButtonState, bool renderButtonState, bool exportButtonState);

private:
    QLineEdit* m_tiffPathField;
    QLineEdit* m_jsonPathField;
    QLineEdit* m_regionField;

    QPushButton* m_generateButton;
    QPushButton* m_renderButton;
    QPushButton* m_exportButton;

    QVTKOpenGLNativeWidget* m_vtkWidget;
    vtkSmartPointer<vtkRenderer> m_renderer;

    std::vector<Mesh> m_generatedMeshes{};

    Renderer* m_meshRenderer;
    ShapefileGenerator* m_shapefileGenerator;
};
