cmake_minimum_required(VERSION 3.16..3.24)

project(TriMeshGenerator    VERSION 0.0.1
                            DESCRIPTION "Delaunay triangulation between 3d points"
                            LANGUAGES CXX)  

add_library(TriMeshGenerator src/triangulation/delaunay.cpp)
add_library(TriMeshGenerator::TriMeshGenerator ALIAS TriMeshGenerator)

target_include_directories(
    TriMeshGenerator 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)

target_link_libraries(
    TriMeshGenerator
    PUBLIC Common::Common
    PRIVATE CGAL::CGAL)
