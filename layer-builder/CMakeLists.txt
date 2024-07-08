cmake_minimum_required(VERSION 3.16..3.24)

project(LayerBuilder    VERSION 0.0.1
                        DESCRIPTION "Layer builder"
                        LANGUAGES CXX)  

add_library(LayerBuilder 
    src/layer_builder.cpp)
add_library(LayerBuilder::LayerBuilder ALIAS LayerBuilder)

target_include_directories(
    LayerBuilder 
        PUBLIC 
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)
find_package(Kriging REQUIRED)
find_package(GeoTiff REQUIRED)
find_package(Normalizer REQUIRED)

target_link_libraries(
    LayerBuilder
        PUBLIC 
            Common::Common
            Kriging::Kriging
            GeoTiff::GeoTiff
            Normalizer::Normalizer
        PRIVATE
            GDAL::GDAL
    )
