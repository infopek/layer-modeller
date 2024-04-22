cmake_minimum_required(VERSION 3.16..3.24)

project(LayerModeller 	VERSION 0.0.1
                        DESCRIPTION "Generating layers from lithostratigraphic data"
                        LANGUAGES CXX)     

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Update submodules
include(cmake/update_submodules.cmake)

# Enable vcpkg
include(${CMAKE_SOURCE_DIR}/external/vcpkg/scripts/buildsystems/vcpkg.cmake)

# Enable testing
include(CTest)

# Before macro, add external dependencies
add_subdirectory(external)

set(as_subproject Common Kriging TriMeshGenerator Renderer App Test)

# Packages
find_package(CGAL CONFIG REQUIRED)
find_package(VTK COMPONENTS 
  CommonCore
  FiltersSources
  RenderingCore
  RenderingOpenGL2
  InteractionStyle
  InteractionWidgets)

find_package(GSL REQUIRED)
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
find_package(nlohmann_json REQUIRED)
include_directories(${GSL_INCLUDE_DIR}) 

# Override default find_package
macro(find_package)
  if(NOT "${ARGV0}" IN_LIST as_subproject)
    message(STATUS "Using native find_package for ${ARGV0}: ${ARGV}")
    _find_package(${ARGV})
  endif()
endmacro()

include_directories(
	PUBLIC
  		vendor)

# Subprojects
add_subdirectory(common)
add_subdirectory(tri-mesh-gen)
add_subdirectory(renderer)
add_subdirectory(interpolator)

add_subdirectory(app)

add_subdirectory(tests)
