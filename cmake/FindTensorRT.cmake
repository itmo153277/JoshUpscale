# variables:
# - TensorRT_ROOT
#   root search directory
# - TensorRT_INCLUDE_DIR
#   header directory
# - TensorRT_LIBRARY_DIR
#   library directory
# - TensorRT_BIN_DIR
#   binary directory (windows only)
#
# sets:
# - TensorRT_FOUND
# - TensorRT_nvinfer_FOUND
# - TensorRT_nvinfer_plugin_FOUND
# - TensorRT_nvparsers_FOUND
# - TensorRT_nvonnxparser_FOUND
# - TensorRT_ROOT (set if absent)
# - TensorRT_INCLUDE_DIR (cached)
# - TensorRT_LIBRARY_DIR (set if absent)
# - TensorRT_BIN_DIR (windows only, cached)
# - TensorRT_nvinfer_LIBRARY (cached)
# - TensorRT_nvinfer_plugin_LIBRARY (cached)
# - TensorRT_nvparsers_LIBRARY (cached)
# - TensorRT_nvonnxparser_LIBRARY (cached)
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
  HINTS ${CUDAToolkit_INCLUDE_DIRS})

if(NOT TensorRT_ROOT)
  get_filename_component(TensorRT_ROOT
    ${TensorRT_INCLUDE_DIR} DIRECTORY)
endif()

if(TensorRT_LIBRARY_DIR)
  find_library(TensorRT_nvinfer_LIBRARY nvinfer
    PATHS ${TensorRT_LIBRARY_DIR} NO_DEFAULT_PATH)
else()
  find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS ${CUDAToolkit_LIBRARY_DIR})
  get_filename_component(TensorRT_LIBRARY_DIR
    ${TensorRT_nvinfer_LIBRARY} DIRECTORY)
endif()

if(WIN32 AND NOT TensorRT_BIN_DIR)
  find_program(TensorRT_BIN_DIR_NVINFER nvinfer.dll
    HINTS ${TensorRT_LIBRARY_DIR}
          ${TensorRT_LIBRARY_DIR}/../bin
          ${CUDAToolkit_BIN_DIR}
    NO_CACHE)
  get_filename_component(TensorRT_BIN_DIR
    ${TensorRT_BIN_DIR_NVINFER} DIRECTORY)
  unset(TensorRT_BIN_DIR_NVINFER)
  set(TensorRT_BIN_DIR "${TensorRT_BIN_DIR}"
    CACHE PATH "TensorRT bin directory")
endif()

find_library(TensorRT_nvinfer_plugin_LIBRARY nvinfer_plugin
  PATHS ${TensorRT_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(TensorRT_nvparsers_LIBRARY nvparsers
  PATHS ${TensorRT_LIBRARY_DIR} NO_DEFAULT_PATH)
find_library(TensorRT_nvonnxparser_LIBRARY nvonnxparser
  PATHS ${TensorRT_LIBRARY_DIR} NO_DEFAULT_PATH)

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

if(TensorRT_nvinfer_LIBRARY)
  set(TensorRT_nvinfer_FOUND 1)
  if(TensorRT_nvinfer_plugin_LIBRARY)
    set(TensorRT_nvinfer_plugin_FOUND 1)
  endif()
  if(TensorRT_nvparsers_LIBRARY)
    set(TensorRT_nvparsers_FOUND 1)
    if(TensorRT_nvonnxparser_LIBRARY)
      set(TensorRT_nvonnxparser_FOUND 1)
    endif()
  endif()
endif()

mark_as_advanced(
  TensorRT_BIN_DIR
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  TensorRT
  REQUIRED_VARS TensorRT_ROOT
                TensorRT_INCLUDE_DIR
                TensorRT_nvinfer_LIBRARY
  VERSION_VAR TensorRT_VERSION
  HANDLE_COMPONENTS
)

mark_as_advanced(
  TensorRT_INCLUDE_DIR
  TensorRT_nvinfer_LIBRARY
  TensorRT_nvinfer_plugin_LIBRARY
  TensorRT_nvparsers_LIBRARY
  TensorRT_nvonnxparser_LIBRARY
)

function(add_tensorrt_lib LIB_NAME)
  add_library(TensorRT::${LIB_NAME} SHARED IMPORTED)
  set_property(TARGET TensorRT::${LIB_NAME}
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${TensorRT_INCLUDE_DIR}")
  if(WIN32)
    set_property(TARGET TensorRT::${LIB_NAME}
      PROPERTY IMPORTED_IMPLIB
      "${TensorRT_${LIB_NAME}_LIBRARY}")
    if(EXISTS "${TensorRT_BIN_DIR}/${LIB_NAME}.dll")
      set_property(TARGET TensorRT::${LIB_NAME}
        PROPERTY IMPORTED_LOCATION
        "${TensorRT_BIN_DIR}/${LIB_NAME}.dll")
    endif()
  else()
    set_property(TARGET TensorRT::${LIB_NAME}
      PROPERTY IMPORTED_LOCATION
      "${TensorRT_${LIB_NAME}_LIBRARY}")
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
