cmake_minimum_required(VERSION 3.16..3.24)

project(App VERSION 0.0.1
            DESCRIPTION "Application"
            LANGUAGES CUDA CXX) 

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

enable_language(CUDA)

add_executable(App 
  src/app.cpp
  src/mainwindow.cpp)

find_package(Common REQUIRED)
find_package(Kriging REQUIRED)
find_package(GeoTiff REQUIRED)
find_package(LayerBuilder REQUIRED)
find_package(MeshGenerator REQUIRED)
find_package(Renderer REQUIRED)

target_link_libraries(
  App 
    PUBLIC
      Common::Common
      Kriging::Kriging
      GeoTiff::GeoTiff
      LayerBuilder::LayerBuilder
      MeshGenerator::MeshGenerator
      Renderer::Renderer
    PRIVATE
      Qt6::Widgets
      CGAL::CGAL
      ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS App
  MODULES ${VTK_LIBRARIES})
