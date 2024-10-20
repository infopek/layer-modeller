#include <shp_generator.h>

#include <converters/cgal_to_vtk_converter.h>

ShapefileGenerator::ShapefileGenerator()
{
}

ShapefileGenerator::~ShapefileGenerator()
{
}

void ShapefileGenerator::addMeshes(const std::vector<Mesh>& meshes)
{
    m_meshes = meshes;

    const size_t numMeshes = meshes.size();
    m_layerBodyPolyData.resize(numMeshes);

    for (size_t i = 0; i < numMeshes; i++)
    {
        const auto& mesh = m_meshes[i];

        auto layerBodyPolyData = CGALToVTKConverter::convertMeshToVTK(mesh.layerBody);

        m_layerBodyPolyData[i] = layerBodyPolyData;
    }
}

void ShapefileGenerator::writeVTKFile(const std::string& path)
{
    int counter = 0;
    for (const auto& mesh : m_meshes)
    {
        vtkSmartPointer<vtkPolyData> polyData = CGALToVTKConverter::convertMeshToVTK(mesh.layerBody);

        // Create a unique filename for each mesh (e.g., baseFilename_0.vtk, baseFilename_1.vtk, ...)
        std::string filename = path + "/layer_" + std::to_string(counter++) + ".vtk";

        // Write the mesh to the file
        vtkSmartPointer<vtkPolyDataWriter> writer = vtkSmartPointer<vtkPolyDataWriter>::New();
        writer->SetFileName(filename.c_str());
        writer->SetInputData(polyData);
        writer->Write();
    }
}