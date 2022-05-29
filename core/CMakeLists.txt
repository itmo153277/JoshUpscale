find_package(CUDAToolkit 11 REQUIRED)
find_package(TensorRT 8.4 REQUIRED COMPONENTS nvinfer)

include(GenerateExportHeader)

file(GLOB_RECURSE SRC src/*.cc src/*.cu)
file(GLOB_RECURSE HEADERS include/*.h public/*.h)
add_library(JoshUpscale SHARED ${SRC} ${HEADERS})
set_target_properties(JoshUpscale PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  CUDA_RUNTIME_LIBRARY Shared
)
generate_export_header(JoshUpscale
  EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/generated/include/JoshUpscale/core/export.h
)
target_include_directories(JoshUpscale PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(JoshUpscale PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/public
  ${CMAKE_CURRENT_BINARY_DIR}/generated/include
)
target_link_libraries(JoshUpscale PRIVATE
  CUDA::cudart
  TensorRT::nvinfer
)

if (ENABLE_LINT)
  if (MSVC)
    set_target_properties(JoshUpscale PROPERTIES
      VS_GLOBAL_RunCodeAnalysis true
      VS_GLOBAL_EnableMicrosoftCodeAnalysis true
      VS_GLOBAL_EnableClangTidyCodeAnalysis true
    )
  endif()
endif()
