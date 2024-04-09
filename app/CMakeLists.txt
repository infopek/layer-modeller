cmake_minimum_required(VERSION 3.16..3.24)
project(App CXX)

add_executable(App main.cpp)

find_package(Common REQUIRED)
find_package(TriMeshGenerator REQUIRED)
find_package(Renderer REQUIRED)

target_link_libraries(
    App 
    PUBLIC Common::Common
    PUBLIC TriMeshGenerator::TriMeshGenerator
    PUBLIC Renderer::Renderer

    PRIVATE CGAL::CGAL
    PRIVATE ${VTK_LIBRARIES}
)

vtk_module_autoinit(
  TARGETS Renderer
  MODULES ${VTK_LIBRARIES}
)
