# variables:
# - TensorRT_ROOT
#   root search directory
# - TensorRT_INCLUDE_DIR
#   header directory
# - TensorRT_LIB_PATH
#   library directory
# - TensorRT_BIN_PATH
#   binary directory (windows only)
#
# sets:
# - TensorRT_FOUND
# - TensorRT_nvinfer_FOUND
# - TensorRT_nvinfer_plugin_FOUND
# - TensorRT_nvparsers_FOUND
# - TensorRT_nvonnxparser_FOUND
# - TensorRT_ROOT
# - TensorRT_INCLUDE_DIR (cached)
# - TensorRT_LIB_PATH
# - TensorRT_BIN_PATH (windows only, cached)
# - TensorRT_LIBRARY_NVINFER (cached)
# - TensorRT_LIBRARY_NVINFER_PLUGIN (cached)
# - TensorRT_LIBRARY_NVPARSERS (cached)
# - TensorRT_LIBRARY_NVONNXPARSER (cached)
# - TensorRT_LIBRARIES
# - TensorRT_VERSION
# - TensorRT_VERSION_MAJOR
# - TensorRT_VERSION_MINOR
# - TensorRT_VERSION_PATCH
#
# targets:
# - TensorRT::nvinfer
# - TensorRT::nvinfer_plugin
# - TensorRT::nvparsers
# - TensorRT::nvonnxparser

find_package(CUDAToolkit QUIET)

find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS ${CUDAToolkit_LIBRARY_ROOT}/include)

if(NOT TensorRT_ROOT)
  get_filename_component(TensorRT_ROOT
    ${TensorRT_INCLUDE_DIR} DIRECTORY)
endif()

if(TensorRT_LIB_PATH)
  find_library(TensorRT_LIBRARY_NVINFER nvinfer
    PATHS ${TensorRT_LIB_PATH} NO_DEFAULT_PATH)
else()
  find_library(TensorRT_LIBRARY_NVINFER nvinfer
    HINTS ${CUDAToolkit_LIBRARY_ROOT}/lib)
  get_filename_component(TensorRT_LIB_PATH
    ${TensorRT_LIBRARY_NVINFER} DIRECTORY)
endif()

if(WIN32 AND NOT TensorRT_BIN_PATH)
  find_program(TensorRT_BIN_PATH_NVINFER nvinfer.dll
    HINTS ${TensorRT_LIB_PATH}
          ${TensorRT_LIB_PATH}/../bin
          ${CUDAToolkit_LIBRARY_ROOT}/bin
    NO_CACHE)
  get_filename_component(TensorRT_BIN_PATH
    ${TensorRT_BIN_PATH_NVINFER} DIRECTORY CACHE)
  unset(TensorRT_BIN_PATH_NVINFER)
endif()

find_library(TensorRT_LIBRARY_NVINFER_PLUGIN nvinfer_plugin
  PATHS ${TensorRT_LIB_PATH} NO_DEFAULT_PATH)
find_library(TensorRT_LIBRARY_NVPARSERS nvparsers
  PATHS ${TensorRT_LIB_PATH} NO_DEFAULT_PATH)
find_library(TensorRT_LIBRARY_NVONNXPARSER nvonnxparser
  PATHS ${TensorRT_LIB_PATH} NO_DEFAULT_PATH)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
    TensorRT_VERSION_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
    TensorRT_VERSION_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h"
    TensorRT_VERSION_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1"
    TensorRT_VERSION_MAJOR "${TensorRT_VERSION_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1"
    TensorRT_VERSION_MINOR "${TensorRT_VERSION_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1"
    TensorRT_VERSION_PATCH "${TensorRT_VERSION_PATCH}")
  string(CONCAT TensorRT_VERSION
    "${TensorRT_VERSION_MAJOR}"
    ".${TensorRT_VERSION_MINOR}"
    ".${TensorRT_VERSION_PATCH}"
  )
endif()

if(TensorRT_LIBRARY_NVINFER)
  set(TensorRT_nvinfer_FOUND 1)
  if(TensorRT_LIBRARY_NVINFER_PLUGIN)
    set(TensorRT_nvinfer_plugin_FOUND 1)
  endif()
  if(TensorRT_LIBRARY_NVPARSERS)
    set(TensorRT_nvparsers_FOUND 1)
    if(TensorRT_LIBRARY_NVONNXPARSER)
      set(TensorRT_nvonnxparser_FOUND 1)
    endif()
  endif()
endif()

mark_as_advanced(
  TensorRT_BIN_PATH
  TensorRT_INCLUDE_DIR
  TensorRT_LIBRARY_NVINFER
  TensorRT_LIBRARY_NVINFER_PLUGIN
  TensorRT_LIBRARY_NVPARSERS
  TensorRT_LIBRARY_NVONNXPARSER
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_ROOT
                TensorRT_INCLUDE_DIR
                TensorRT_LIBRARY_NVINFER
  VERSION_VAR TensorRT_VERSION
  HANDLE_COMPONENTS
)

function(add_tensorrt_lib LIB_NAME)
  string(TOUPPER ${LIB_NAME} LIB_NAME_UPPER)
  add_library(TensorRT::${LIB_NAME} SHARED IMPORTED)
  set_property(TARGET TensorRT::${LIB_NAME}
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${TensorRT_INCLUDE_DIR}")
  if(WIN32)
    set_property(TARGET TensorRT::${LIB_NAME}
      PROPERTY IMPORTED_IMPLIB
      "${TensorRT_LIBRARY_${LIB_NAME_UPPER}}")
    if(EXISTS "${TensorRT_BIN_PATH}/${LIB_NAME}.dll")
      set_property(TARGET TensorRT::${LIB_NAME}
        PROPERTY IMPORTED_LOCATION
        "${TensorRT_BIN_PATH}/${LIB_NAME}.dll")
    endif()
  else()
    set_property(TARGET TensorRT::${LIB_NAME} 
      PROPERTY IMPORTED_LOCATION
      "${TensorRT_LIBRARY_${LIB_NAME_UPPER}}")
  endif()
  if(${LIB_NAME} IN_LIST TensorRT_FIND_COMPONENTS)
    list(APPEND TensorRT_LIBRARIES TensorRT::${LIB_NAME})
    set(TensorRT_LIBRARIES "${TensorRT_LIBRARIES}" PARENT_SCOPE)
  endif()
endfunction()

set(TensorRT_LIBRARIES "")
if(TensorRT_FOUND)
  foreach(LIB_NAME nvinfer nvinfer_plugin nvparsers nvonnxparser)
    if(TensorRT_${LIB_NAME}_FOUND)
      add_tensorrt_lib(${LIB_NAME})
    endif()
  endforeach()
endif()
