cmake_minimum_required(VERSION 3.16..3.24)

project(GeoTiff    VERSION 0.0.1
                    DESCRIPTION "GeoTIFF module"
                    LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Enable CUDA language
enable_language(CUDA)

add_library(GeoTiff
    src/blur/blur.cu
    src/blur/blur.cpp)
add_library(GeoTiff::GeoTiff ALIAS GeoTiff)

set_target_properties(GeoTiff PROPERTIES CUDA_ARCHITECTURES all-major)

target_include_directories(GeoTiff
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link libraries
target_link_libraries(GeoTiff 
    PRIVATE 
        GDAL::GDAL
        ${CUDA_LIBRARIES}
        ${CUDA_CUFFT_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${CUDA_cupti_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_cusparse_LIBRARY}
        ${CUDA_npp_LIBRARY}
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppi_LIBRARY} 
        ${CUDA_npps_LIBRARY}
        ${CUDA_nvcuvenc_LIBRARY} ${CUDA_nvcuvid_LIBRARY}
        opencv_ml 
        opencv_dnn 
        opencv_core
        opencv_imgcodecs 
        opencv_flann)