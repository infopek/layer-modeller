cmake_minimum_required(VERSION 3.27)

project(Common  VERSION 0.0.1
            	DESCRIPTION "Common project for headers"
                LANGUAGES CXX)  

add_library(Common 
  	src/src.cpp
	src/logging.cpp
	src/cgal_to_vtk_converter.cpp)
add_library(Common::Common ALIAS Common)

target_include_directories(
    Common 
		PUBLIC 		
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/vendor/perlin-noise>
            $<INSTALL_INTERFACE:include>
	)

target_link_libraries(
    Common
    	PRIVATE 
			CGAL::CGAL
    		${VTK_LIBRARIES})

vtk_module_autoinit(
	TARGETS Common
	MODULES ${VTK_LIBRARIES})
