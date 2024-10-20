cmake_minimum_required(VERSION 3.27)

project(ShpGenerator	VERSION 0.0.1
                    DESCRIPTION "Shapefile generator using VTK"
                    LANGUAGES CXX)  

add_library(ShpGenerator
  	src/shp_generator.cpp)
add_library(ShpGenerator::ShpGenerator ALIAS ShpGenerator)

target_include_directories(
    ShpGenerator 
		PUBLIC 	
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)
find_package(MeshGenerator REQUIRED)

target_link_libraries(
    ShpGenerator
        PUBLIC
            Common::Common
            MeshGenerator::MeshGenerator
        PRIVATE 
            ${VTK_LIBRARIES})

vtk_module_autoinit(
  	TARGETS ShpGenerator
  	MODULES ${VTK_LIBRARIES})