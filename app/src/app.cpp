#include <geotiff_handler.h>
#include <layer_builder.h>
#include <modeller/modeller_set.h>
#include <renderer.h>
#include <blur/blur.h>

#include "mainwindow.h"
#include <QApplication>

#include <iostream>
#include <string>

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QPushButton>
#include <QWidget>
#include <QVTKOpenGLNativeWidget.h>
#include <vtkSmartPointer.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>

class CustomVTKWidget : public QVTKOpenGLNativeWidget
{
public:
    CustomVTKWidget(QWidget* parent = nullptr) : QVTKOpenGLNativeWidget(parent) {}

    vtkRenderWindow* GetRenderWindow() const
    {
        return this->renderWindow();
    }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow()
    {
        // Create the main widget
        QWidget* mainWidget = new QWidget;
        QVBoxLayout* layout = new QVBoxLayout;

        // Create the text field and button
        QLineEdit* textField = new QLineEdit;
        QPushButton* button = new QPushButton("Display Text");

        layout->addWidget(textField);
        layout->addWidget(button);

        // Create the custom VTK widget
        CustomVTKWidget* vtkWidget = new CustomVTKWidget;
        layout->addWidget(vtkWidget);

        mainWidget->setLayout(layout);
        setCentralWidget(mainWidget);

        // Setup VTK
        vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
        vtkWidget->GetRenderWindow()->AddRenderer(renderer);

        vtkSmartPointer<vtkTextActor> textActor = vtkSmartPointer<vtkTextActor>::New();
        renderer->AddActor(textActor);

        connect(button, &QPushButton::clicked, [textField, textActor, vtkWidget]() {
            QString text = textField->text();
            textActor->SetInput(text.toStdString().c_str());
            textActor->GetTextProperty()->SetFontSize(24);
            textActor->SetPosition(10, 10);
            vtkWidget->GetRenderWindow()->Render();
            });
    }
};

int main(int argc, char* argv[])
{
    QCoreApplication::addLibraryPath("../../vcpkg_installed/x64-windows/debug/Qt6/plugins");

    QApplication app(argc, argv);

    MainWindow window;
    window.show();

    return app.exec();
}

#include "app.moc"

// int main(int argc, char* argv[])
// {
//     // Get input
//     QApplication app(argc, argv);
//     MainWindow window;
//     window.show();
//     app.exec();

//     const std::string region = window.getRegion();
//     const std::string observationDataPath = window.getJsonPath();
//     const std::string tiffPath = window.getTiffPath();

//     // Show output
//     // LayerBuilder layerBuilder(region, observationDataPath, tiffPath);

//     // ModellerSet modeller(layerBuilder);
//     // modeller.createMeshes();

//     // Renderer renderer{};
//     // renderer.addMeshes(modeller.getMeshes());

//     // // Describe what you want to be rendered
//     // renderer.prepareEdges();
//     // renderer.prepareSurfaces();
//     // renderer.prepareLayerBody();

//     // // Render
//     // renderer.render();
// }
