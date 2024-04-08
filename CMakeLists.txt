set(PROJECT_NAME_STR "DelaunayTriangulation")
set(CMAKE_CXX_STANDARD 20)

cmake_minimum_required(VERSION 3.16..3.24)
project(${PROJECT_NAME_STR} VERSION 0.0.1
                            DESCRIPTION "Delaunay triangulation with vtk"
                            LANGUAGES CXX)     

find_package(CGAL CONFIG REQUIRED)
find_package(VTK COMPONENTS 
  CommonCore
  FiltersSources
  RenderingCore
  RenderingOpenGL2
  InteractionStyle
  InteractionWidgets
)

include_directories(
  vendor
)

file(GLOB_RECURSE SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME_STR} ${SOURCES})

target_link_libraries(${PROJECT_NAME_STR}
  PRIVATE
  CGAL::CGAL
  ${VTK_LIBRARIES}
)

vtk_module_autoinit(
  TARGETS ${PROJECT_NAME_STR}
  MODULES ${VTK_LIBRARIES}
)