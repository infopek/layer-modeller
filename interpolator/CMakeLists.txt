cmake_minimum_required(VERSION 3.27)

project(Kriging VERSION 0.0.1
                DESCRIPTION "Krigin interpolation for pseudo-layer generator"
                LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# Enable CUDA language
enable_language(CUDA)

add_library(Kriging
    src/interpolator.cpp
    src/kriging_cpu.cpp
    src/variogram.cpp
    src/initialization.cpp
    src/plotting.cpp
)
add_library(Kriging::Kriging ALIAS Kriging)

find_package(Common REQUIRED)

set_target_properties(Kriging PROPERTIES CUDA_ARCHITECTURES all-major)

target_include_directories(Kriging
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link libraries
target_link_libraries(Kriging 
    PUBLIC 
        Common::Common
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
        ${CUDA_nvcuvenc_LIBRARY} ${CUDA_nvcuvid_LIBRARY}
        GSL::gsl 
        GSL::gslcblas 
        nlohmann_json::nlohmann_json
        Eigen3::Eigen3)

