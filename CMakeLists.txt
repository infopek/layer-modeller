cmake_minimum_required(VERSION 3.16)

project(LayerModeller VERSION 0.0.1
                      DESCRIPTION "Generating layers from lithostratigraphic data"
                      LANGUAGES CXX)     

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Update submodules
include(cmake/update_submodules.cmake) 

# Enable vcpkg
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)
  message(FATAL_ERROR "Vcpkg integration file not found. Please ensure the path is correct.")
endif()
include(${CMAKE_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

# Enable testing
include(CTest)

# Before macro, add external dependencies
add_subdirectory(external)

set(as_subproject Common Kriging GeoTiff LayerBuilder MeshGenerator Renderer Normalizer App)

# Packages
find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(VTK COMPONENTS 
  CommonCore
  CommonDataModel
  FiltersCore
  # FiltersGeneral
  # FiltersSources
  # FiltersGeometry
  # FiltersModeling
  FiltersExtraction
  FiltersSources
  # InteractionStyle
  # InteractionWidgets
  RenderingCore
  RenderingOpenGL2
  # IOXML
  GUISupportQt
  REQUIRED)
find_package(GSL REQUIRED)
find_package(GDAL CONFIG REQUIRED)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
find_package(nlohmann_json REQUIRED)

# For QT6
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Override default find_package
macro(find_package)
  if(NOT "${ARGV0}" IN_LIST as_subproject)
    message(STATUS "Using native find_package for ${ARGV0}: ${ARGV}")
    _find_package(${ARGV})
  endif()
endmacro()

# Subprojects
add_subdirectory(common)
add_subdirectory(geo-tiff)
add_subdirectory(normalization)
add_subdirectory(interpolator)
add_subdirectory(layer-builder)
add_subdirectory(mesh-generator)
add_subdirectory(renderer)
add_subdirectory(app)
# add_subdirectory(tests)