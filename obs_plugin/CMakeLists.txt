find_package(FFMPEG REQUIRED)
find_package(libobs QUIET)
if(NOT TARGET OBS::libobs)
  find_package(LibObs REQUIRED)
  add_library(OBS::libobs ALIAS libobs)
  set(libobs_VERSION ${LibObs_VERSION})
endif()

file(GLOB_RECURSE SRC src/*.c src/*.cc)
file(GLOB_RECURSE HEADERS include/*.h)
add_library(obs-joshupscale SHARED ${SRC} ${HEADERS})
set_target_properties(obs-joshupscale PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
)
if(NOT MSVC)
  target_link_options(obs-joshupscale PRIVATE -Wl,--exclude-libs,ALL)
endif()
target_compile_features(obs-joshupscale PUBLIC cxx_std_20)
target_include_directories(obs-joshupscale PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(obs-joshupscale SYSTEM PRIVATE
  ${FFMPEG_INCLUDE_DIRS}
)
if (WIN32)
  option(USE_NVVFX "Use NVIDIA Video Effects for OBS plugin" ON)
  if(USE_NVVFX)
    find_package(CUDAToolkit 11 REQUIRED)
    find_package(TensorRT 8.4 REQUIRED COMPONENTS nvinfer)
    if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 11)
      message(FATAL_ERROR "Need CUDA 11 to use NVVFX, found ${CUDAToolkit_VERSION}")
    endif()
    if(NOT (TensorRT_VERSION_MAJOR EQUAL 8 AND TensorRT_VERSION_MINOR EQUAL 4))
      message(FATAL_ERROR "Need TensorRT 8.4 to use NVVFX, found ${TensorRT_VERSION}")
    endif()
    target_compile_definitions(obs-joshupscale PRIVATE JOSHUPSCALE_NVVFX)
  endif()
endif()
target_link_directories(obs-joshupscale PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(obs-joshupscale PRIVATE
  JoshUpscale
  OBS::libobs
  ${FFMPEG_LIBRARIES}
)
if(MSVC)
  target_link_options(obs-joshupscale PRIVATE /DELAYLOAD:JoshUpscale.dll)
endif()

if(BUILD_OBS_PLUGIN_INSTALLER)
  find_program(ISCC_COMMAND "iscc"
    HINTS "$ENV{ProgramFiles}/Inno Setup 6"
    REQUIRED
  )
  set(OBS_PLUGIN_REQUIRED_DLLS "" CACHE STRING "Required dlls for OBS plugin")
  add_custom_command(
    OUTPUT obs-joshupscale-installer.iss
    MAIN_DEPENDENCY install/obs-joshupscale-installer.iss.in
    DEPENDS JoshUpscale obs-joshupscale
    COMMAND ${CMAKE_COMMAND}
      -DTEMPLATE_IN=${CMAKE_CURRENT_SOURCE_DIR}/install/obs-joshupscale-installer.iss.in
      -DTEMPLATE_OUT=${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss
      -DPROJECT_VERSION=${PROJECT_VERSION}
      -DDATA_PATH=${PROJECT_SOURCE_DIR}/data
      -DLICENSE_PATH=${PROJECT_SOURCE_DIR}/LICENSE
      -DCORE_PATH=$<TARGET_FILE:JoshUpscale>
      -DPLUGIN_PATH=$<TARGET_FILE:obs-joshupscale>
      -DUSE_NVVFX=$<IF:$<BOOL:USE_NVVFX>,True,False>
      -DREQUIRED_DLLS=${OBS_PLUGIN_REQUIRED_DLLS}
      -DMIN_OBS_VERSION=$<IF:$<VERSION_LESS:${libobs_VERSION},28.0.0>,0,28>
      -P ${PROJECT_SOURCE_DIR}/cmake/GenerateTemplate.cmake
    COMMAND ${CMAKE_COMMAND}
      -E touch_nocreate ${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss
  )
  file(GLOB_RECURSE DATA ${PROJECT_SOURCE_DIR}/data/*)
  if(NOT DATA)
    message(FATAL_ERROR "No model data found. Please put data into ./data folder")
  endif()
  add_custom_command(
    OUTPUT obs-joshupscale-installer.exe
    COMMAND ${ISCC_COMMAND}
      /Q "/O${CMAKE_CURRENT_BINARY_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss"
    COMMENT "Building installer"
    DEPENDS obs-joshupscale-installer.iss JoshUpscale obs-joshupscale ${DATA}
  )
  add_custom_target(obs-joshupscale-installer-target ALL
    DEPENDS obs-joshupscale-installer.exe
    SOURCES install/obs-joshupscale-installer.iss.in
  )
endif()
