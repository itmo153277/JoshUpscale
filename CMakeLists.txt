cmake_minimum_required(VERSION 3.12)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
project(JoshUpscale VERSION 1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

option(ENABLE_LINT "Enable linting" OFF)
if(ENABLE_LINT)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /external:anglebrackets /external:W0 /wd26495 /wd28251")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/W4 -Werror all-warnings")
    list(APPEND CMAKE_VS_GLOBALS
      "RunCodeAnalysis=true"
      "EnableMicrosoftCodeAnalysis=true"
      "EnableClangTidyCodeAnalysis=true"
    )
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wconversion -Werror")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wconversion -Werror all-warnings")
    find_program(CLANG_TIDY "clang-tidy" REQUIRED)
    find_program(CPPLINT "cpplint" REQUIRED)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY};--warnings-as-errors=*")
    set(CMAKE_CXX_CPPLINT "${CPPLINT};--quiet")
  endif()
endif()

add_subdirectory(core)
if(WIN32)
  add_subdirectory(avisynth_plugin)
endif()
