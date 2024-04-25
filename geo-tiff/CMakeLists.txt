cmake_minimum_required(VERSION 3.16..3.24)

project(GeoTiff     VERSION 0.0.1
                    DESCRIPTION "Delaunay triangulation between 3d points"
                    LANGUAGES CXX)  

add_library(GeoTiff src/tiff/tiff.cpp)
add_library(GeoTiff::GeoTiff ALIAS GeoTiff)

target_include_directories(
    GeoTiff 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)
find_package(Blur REQUIRED)

target_link_libraries(
    GeoTiff
        PUBLIC 
            Common::Common
            Blur::Blur
        PRIVATE 
            CGAL::CGAL)