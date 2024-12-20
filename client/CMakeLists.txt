cmake_minimum_required(VERSION 3.27)

project(RestClient VERSION 1.0
                DESCRIPTION "C++ Client Module to access layer modeller server"
                LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_library(RestClient
    src/client.cpp
    src/zip-handler.cpp 
)
add_library(RestClient::RestClient ALIAS RestClient)
target_include_directories(RestClient
    PRIVATE
        ${CURL_INCLUDE_DIRS}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

target_link_libraries(RestClient 
    PRIVATE
        libzip::zip 
        ZLIB::ZLIB
        CURL::libcurl
        ${CURL_LIBRARIES}
)
