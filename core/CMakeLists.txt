option(USE_LEAN_RT "Link against lean runtime" OFF)
if(USE_LEAN_RT)
  set(NVINFER_LIB nvinfer_lean)
else()
  set(NVINFER_LIB nvinfer)
endif()
option(USE_STATIC_CUDA "Link against static CUDA runtime" OFF)
if(USE_STATIC_CUDA)
  set(CUDART_LIB Static)
else()
  set(CUDART_LIB Shared)
endif()

find_package(CUDAToolkit 11 REQUIRED)
find_package(TensorRT 8.5 REQUIRED COMPONENTS ${NVINFER_LIB})

include(GenerateExportHeader)

file(GLOB_RECURSE SRC src/*.cc src/*.cu)
file(GLOB_RECURSE HEADERS include/*.h public/*.h)
add_library(JoshUpscale SHARED ${SRC} ${HEADERS})
set_target_properties(JoshUpscale PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  CUDA_RUNTIME_LIBRARY ${CUDART_LIB}
)
if(NOT MSVC)
  target_link_options(JoshUpscale PRIVATE -Wl,--exclude-libs,ALL)
endif()
generate_export_header(JoshUpscale
  EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/include/JoshUpscale/core/export.h
)
target_compile_features(JoshUpscale PUBLIC cxx_std_20 cuda_std_20)
target_include_directories(JoshUpscale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(JoshUpscale PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/public
  ${CMAKE_CURRENT_BINARY_DIR}/generated/include
)
target_link_libraries(JoshUpscale PRIVATE
  CUDA::$<IF:$<BOOL:${USE_STATIC_CUDA}>,cudart_static,cudart>
  TensorRT::${NVINFER_LIB}
)
