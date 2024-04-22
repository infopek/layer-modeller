cmake_minimum_required(VERSION 3.16..3.24)

project(App VERSION 0.0.1
            DESCRIPTION "Application"
            LANGUAGES CXX) 

add_executable(App application.cpp)

find_package(Common REQUIRED)
find_package(TriMeshGenerator REQUIRED)
find_package(Renderer REQUIRED)

target_link_libraries(
  App 
    PUBLIC
      Common::Common
      Kriging::Kriging
      TriMeshGenerator::TriMeshGenerator
      Renderer::Renderer
    PRIVATE
      CGAL::CGAL
      ${VTK_LIBRARIES})

vtk_module_autoinit(
  TARGETS Renderer
  MODULES ${VTK_LIBRARIES})
