cmake_minimum_required(VERSION 3.21)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WIN32)
  option(BUILD_AVISYNTH_PLUGIN "Build AviSynth plugin" ON)
endif()
if(BUILD_AVISYNTH_PLUGIN)
  list(APPEND VCPKG_MANIFEST_FEATURES "avisynth-plugin")
endif()
option(BUILD_OBS_PLUGIN "Build OBS plugin" OFF)
if(BUILD_OBS_PLUGIN)
  list(APPEND VCPKG_MANIFEST_FEATURES "obs-plugin")
  if(WIN32)
    option(BUILD_OBS_PLUGIN_INSTALLER "Build OBS plugin installer" ON)
  endif()
endif()

project(JoshUpscale VERSION 2.0 LANGUAGES C CXX CUDA)

include(GNUInstallDirs)
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string(REPLACE ${PROJECT_NAME} ${PROJECT_NAME_LOWER} CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DOCDIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/JoshUpscaleConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/JoshUpscaleConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JoshUpscale
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/JoshUpscaleConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
endif()

option(ENABLE_LINT "Enable linting" OFF)
if(ENABLE_LINT)
  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /external:anglebrackets /external:W0 /wd26439 /wd26495")
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
    set(CMAKE_CUDA_CLANG_TIDY "${CLANG_TIDY};--warnings-as-errors=*")
    set(CMAKE_C_CLANG_TIDY "${CLANG_TIDY};--warnings-as-errors=*")
    set(CMAKE_CXX_CPPLINT "${CPPLINT};--quiet")
    set(CMAKE_CUDA_CPPLINT "${CPPLINT};--quiet")
    set(CMAKE_C_CPPLINT "${CPPLINT};--quiet")
  endif()
endif()

add_subdirectory(core)
if(BUILD_AVISYNTH_PLUGIN)
  add_subdirectory(avisynth_plugin)
endif()
if(BUILD_OBS_PLUGIN)
  add_subdirectory(obs_plugin)
endif()

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/JoshUpscaleConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/JoshUpscaleConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JoshUpscale
  COMPONENT Development
)

export(
  EXPORT JoshUpscaleTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/JoshUpscaleTargets.cmake
  NAMESPACE JoshUpscale::
)

install(
  EXPORT JoshUpscaleTargets
  NAMESPACE JoshUpscale::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/JoshUpscale
  COMPONENT Development
)

if(LINUX AND EXISTS "/etc/debian_version")
  install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    RENAME copyright
    COMPONENT Runtime
  )
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  set(CPACK_GENERATOR "DEB")
  set(CPACK_DEB_COMPONENT_INSTALL ON)
  set(CPACK_STRIP_FILES ON)
  set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Runtime Package for JoshUpscale\nAI video upscaler library runtime package")
  set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Development Package for JoshUpscale\nAI video upscaler library dev package")
  set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS Runtime)
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Viktor Ivanov <viktprog@gmail.com>")
  set(CPACK_RESOURCE_FILE_LICENSE ${PROJECT_SOURCE_DIR}/LICENSE)
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_NAME ${PROJECT_NAME})
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_SECTION libs)
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_SHLIBDEPS ON)
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/cpack/shlibs "libJoshUpscale ${PROJECT_VERSION} joshupscale (= ${PROJECT_VERSION})")
  set(CPACK_DEBIAN_RUNTIME_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_BINARY_DIR}/cpack/shlibs")
  if(BUILD_OBS_PLUGIN)
    set(CPACK_DEBIAN_RUNTIME_PACKAGE_RECOMMENDS obs-studio)
  endif()
  set(CPACK_DEBIAN_DEVELOPMENT_PACKAGE_NAME "${PROJECT_NAME}-dev")
  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)
  include(CPack)
endif()
