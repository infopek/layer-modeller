cmake_minimum_required(VERSION 3.16..3.24)

project(Common  VERSION 0.0.1
            	DESCRIPTION "Common project for headers"
                LANGUAGES CXX)  

add_library(Common 
  	src/common-includes/cgal.cpp
  	src/common-includes/vtk.cpp)
add_library(Common::Common ALIAS Common)

target_include_directories(
    Common 
		PUBLIC 		
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
