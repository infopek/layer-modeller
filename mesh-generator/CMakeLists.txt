cmake_minimum_required(VERSION 3.16..3.24)

project(MeshGenerator   VERSION 0.0.1
                        DESCRIPTION "Mesh generator"
                        LANGUAGES CXX)  

add_library(MeshGenerator 
    src/modeller/modeller_set.cpp)
add_library(MeshGenerator::MeshGenerator ALIAS MeshGenerator)

target_include_directories(
    MeshGenerator 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)
find_package(LayerBuilder REQUIRED)

target_link_libraries(
    MeshGenerator
    PUBLIC
        Common::Common
        LayerBuilder::LayerBuilder
    PRIVATE 
        CGAL::CGAL)
