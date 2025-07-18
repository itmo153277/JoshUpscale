find_package(libobs QUIET)
if(NOT TARGET OBS::libobs)
  find_package(LibObs REQUIRED)
  add_library(OBS::libobs ALIAS libobs)
  set(libobs_VERSION ${LibObs_VERSION})
endif()

file(GLOB_RECURSE SRC src/*.c src/*.cc)
file(GLOB_RECURSE HEADERS include/*.h)
file(GLOB_RECURSE DATA data/*)
file(GLOB_RECURSE DATA_MODELS data/*.trt)
if(NOT DATA_MODELS)
  message(FATAL_ERROR "No models found. Please put your models into obs_plugin/data folder")
endif()
foreach(F IN LISTS DATA_MODELS)
  list(REMOVE_ITEM DATA ${F})
endforeach()
add_library(obs-joshupscale SHARED ${SRC} ${HEADERS} ${DATA})
source_group(Resources FILES ${DATA})
set_target_properties(obs-joshupscale PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  INSTALL_RPATH "$ORIGIN/.."
)
if(NOT MSVC)
  target_link_options(obs-joshupscale PRIVATE -Wl,--exclude-libs,ALL)
endif()
target_compile_features(obs-joshupscale PUBLIC cxx_std_20)
target_include_directories(obs-joshupscale PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(WIN32)
  option(USE_NVVFX "Use NVIDIA Video Effects for OBS plugin" ON)
  if(USE_NVVFX)
    if(USE_STATIC_CUDA OR USE_LEAN_RT OR USE_PRIVATE_DLLS)
      message(FATAL_ERROR "Invalid config")
    endif()
    find_package(CUDAToolkit 12 QUIET REQUIRED)
    find_package(TensorRT 10 QUIET REQUIRED COMPONENTS nvinfer)
    if(NOT CUDAToolkit_VERSION_MAJOR EQUAL 12)
      message(FATAL_ERROR "Need CUDA 12 to use NVVFX, found ${CUDAToolkit_VERSION}")
    endif()
    if(NOT TensorRT_VERSION_MAJOR EQUAL 10)
      message(FATAL_ERROR "Need TensorRT 10 to use NVVFX, found ${TensorRT_VERSION}")
    endif()
    target_compile_definitions(obs-joshupscale PRIVATE JOSHUPSCALE_NVVFX)
  endif()
endif()
target_link_libraries(obs-joshupscale PRIVATE
  JoshUpscale
  OBS::libobs
)
if(MSVC)
  target_link_options(obs-joshupscale PRIVATE /DELAYLOAD:JoshUpscale.dll)
endif()

if(BUILD_OBS_PLUGIN_INSTALLER)
  find_program(ISCC_COMMAND "iscc"
    HINTS "$ENV{ProgramFiles}/Inno Setup 6"
          "$ENV{ProgramFiles\(x86\)}/Inno Setup 6"
    REQUIRED
  )
  # 25..27..28..30..
  set(MIN_OBS_VERSION 25)
  if(libobs_VERSION_MAJOR GREATER_EQUAL 30)
    set(MIN_OBS_VERSION ${libobs_VERSION_MAJOR})
  elseif(libobs_VERSION_MAJOR GREATER_EQUAL 28)
    set(MIN_OBS_VERSION 28)
  elseif(libobs_VERSION_MAJOR GREATER_EQUAL 27)
    set(MIN_OBS_VERSION 27)
  endif()
  add_custom_command(
    OUTPUT obs-joshupscale-installer.iss
    MAIN_DEPENDENCY install/obs-joshupscale-installer.iss.in
    DEPENDS JoshUpscale obs-joshupscale
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_CURRENT_BINARY_DIR}/dlls
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/dlls
    COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:${USE_NVVFX}>,true,copy>
      -t ${CMAKE_CURRENT_BINARY_DIR}/dlls $<TARGET_RUNTIME_DLLS:JoshUpscale>
    COMMAND ${CMAKE_COMMAND}
      -DTEMPLATE_IN=${CMAKE_CURRENT_SOURCE_DIR}/install/obs-joshupscale-installer.iss.in
      -DTEMPLATE_OUT=${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss
      -DPROJECT_VERSION=${PROJECT_VERSION}
      -DDATA_PATH=${CMAKE_CURRENT_SOURCE_DIR}/data
      -DLICENSE_PATH=${PROJECT_SOURCE_DIR}/LICENSE
      -DCORE_PATH=$<TARGET_FILE:JoshUpscale>
      -DPLUGIN_PATH=$<TARGET_FILE:obs-joshupscale>
      -DUSE_NVVFX=$<IF:$<BOOL:${USE_NVVFX}>,True,False>
      -DDEPS_PATH=${CMAKE_CURRENT_BINARY_DIR}/dlls
      -DMIN_OBS_VERSION=${MIN_OBS_VERSION}
      -P ${PROJECT_SOURCE_DIR}/cmake/GenerateTemplate.cmake
    COMMAND ${CMAKE_COMMAND}
      -E touch_nocreate ${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss
  )
  add_custom_command(
    OUTPUT obs-joshupscale-installer.exe
    COMMAND ${ISCC_COMMAND}
      /Q "/O${CMAKE_CURRENT_BINARY_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/obs-joshupscale-installer.iss"
    COMMENT "Building installer"
    DEPENDS
      obs-joshupscale-installer.iss
      JoshUpscale
      obs-joshupscale
      ${DATA}
      ${DATA_MODELS}
  )
  add_custom_target(obs-joshupscale-installer-target ALL
    DEPENDS obs-joshupscale-installer.exe
    SOURCES install/obs-joshupscale-installer.iss.in
  )
endif()
set_target_properties(obs-joshupscale PROPERTIES NO_SONAME ON)

install(
  TARGETS obs-joshupscale
  RUNTIME
    COMPONENT Runtime
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/obs-plugins
  LIBRARY
    COMPONENT Runtime
    NAMELINK_SKIP
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/obs-plugins
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/obs/obs-plugins/obs-joshupscale
  COMPONENT Runtime
)
