cmake_minimum_required(VERSION 3.27)

project(Renderer	VERSION 0.0.1
                    DESCRIPTION "Renderer using VTK"
                    LANGUAGES CXX)  

add_library(Renderer 
  	src/renderer.cpp)
add_library(Renderer::Renderer ALIAS Renderer)

target_include_directories(
    Renderer 
		PUBLIC 	
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)
find_package(MeshGenerator REQUIRED)

target_link_libraries(
    Renderer
        PUBLIC
            Common::Common
            MeshGenerator::MeshGenerator
        PRIVATE 
            ${VTK_LIBRARIES})

vtk_module_autoinit(
  	TARGETS Renderer
  	MODULES ${VTK_LIBRARIES})