cmake_minimum_required(VERSION 3.16..3.24)

project(Normalizer	VERSION 0.0.1
                    DESCRIPTION "Renderer using VTK"
                    LANGUAGES CXX)  

add_library(Normalizer 
  	src/normalizer.cpp)
add_library(Normalizer::Normalizer ALIAS Normalizer)

target_include_directories(
    Normalizer 
		PUBLIC 	
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)

find_package(Common REQUIRED)

target_link_libraries(
    Normalizer
        PUBLIC
            Common::Common
        )
