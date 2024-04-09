cmake_minimum_required(VERSION 3.16..3.24)

project(LayerModeller    VERSION 0.0.1
                         DESCRIPTION "Generating layers from lithostratigraphic data"
                         LANGUAGES CXX)     

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(as_subproject Common TriMeshGenerator Renderer App)

find_package(CGAL CONFIG REQUIRED)
find_package(VTK COMPONENTS 
  CommonCore
  FiltersSources
  RenderingCore
  RenderingOpenGL2
  InteractionStyle
  InteractionWidgets
)

macro(find_package)
  if(NOT "${ARGV0}" IN_LIST as_subproject)
    message(STATUS "Using native find_package for ${ARGV0}: ${ARGV}")
    _find_package(${ARGV})
  endif()
endmacro()

include_directories(
  vendor
)

add_subdirectory(common)
add_subdirectory(tri-mesh-gen)
add_subdirectory(renderer)
add_subdirectory(app)
