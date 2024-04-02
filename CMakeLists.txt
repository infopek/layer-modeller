cmake_minimum_required(VERSION 3.14)

set(PROJECT_NAME_STR "GDALTest")

project(${PROJECT_NAME_STR})

find_package(GDAL CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(VTK COMPONENTS 
  CommonColor
  CommonCore
  CommonDataModel
  FiltersCore
  FiltersSources
  IOImage
  IOGeometry
  IOLegacy
  IOPLY
  IOXML
  ImagingSources
  InteractionImage
  InteractionStyle
  InteractionWidgets
  RenderingContextOpenGL2
  RenderingCore
  RenderingFreeType
  RenderingGL2PSOpenGL2
  RenderingOpenGL2
)

include("${VTK_USE_FILE}")

add_executable(${PROJECT_NAME_STR} src/source.cpp)

target_link_libraries(${PROJECT_NAME_STR}
  PRIVATE
  GDAL::GDAL
  CGAL::CGAL
  ${VTK_LIBRARIES}
)

vtk_module_autoinit(
  TARGETS ${PROJECT_NAME_STR}
  MODULES ${VTK_LIBRARIES}
)