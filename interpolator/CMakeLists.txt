cmake_minimum_required(VERSION 3.16..3.24)

project(Kriging VERSION 0.0.1
                DESCRIPTION "Krigin interpolation for pseudo-layer generator"
                LANGUAGES CUDA CXX)

# Enable CUDA language
enable_language(CUDA)

add_library(Kriging
    src/kernel.cu
    src/kriging_cpu.cpp
    src/kriging_gpu.cu
    src/kriging_utilities.cu
    src/variogram.cu
    src/initialization.cpp
    src/plotting.cpp)
add_library(Kriging::Kriging ALIAS Kriging)

set_target_properties(Kriging PROPERTIES CUDA_ARCHITECTURES all-major)

target_include_directories(Kriging
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link libraries
target_link_libraries(Kriging 
    PRIVATE 
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
        ${CUDA_nvcuvenc_LIBRARY} ${CUDA_nvcuvid_LIBRARY} GSL::gsl GSL::gslcblas nlohmann_json::nlohmann_json)