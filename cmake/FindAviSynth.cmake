# variables:
# - AviSynth_ROOT
#   root search directory
# - AviSynth_INCLUDE_DIR
#   header directory
#
# sets:
# - AviSynth_FOUND
# - AviSynth_INCLUDE_DIR (cached)
# - AviSynth_VERSION
# - AviSynth_VERSION_MAJOR
# - AviSynth_VERSION_MINOR
# - AviSynth_VERSION_PATCH
#
# targets:
# - AviSynth

find_path(AviSynth_INCLUDE_DIR avisynth/avisynth.h
  PATH_SUFFIXES include)

if(AviSynth_INCLUDE_DIR AND
  EXISTS "${AviSynth_INCLUDE_DIR}/avisynth/avs/version.h")
  file(STRINGS "${AviSynth_INCLUDE_DIR}/avisynth//avs/version.h"
    AviSynth_VERSION_MAJOR REGEX "^#define +AVS_MAJOR_VER +[0-9]+.*$")
  file(STRINGS "${AviSynth_INCLUDE_DIR}/avisynth//avs/version.h"
    AviSynth_VERSION_MINOR REGEX "^#define +AVS_MINOR_VER +[0-9]+.*$")
  file(STRINGS "${AviSynth_INCLUDE_DIR}/avisynth//avs/version.h"
    AviSynth_VERSION_PATCH REGEX "^#define +AVS_BUGFIX_VER +[0-9]+.*$")

  string(REGEX REPLACE "^#define +AVS_MAJOR_VER +([0-9]+).*$" "\\1"
    AviSynth_VERSION_MAJOR "${AviSynth_VERSION_MAJOR}")
  string(REGEX REPLACE "^#define +AVS_MINOR_VER +([0-9]+).*$" "\\1"
    AviSynth_VERSION_MINOR "${AviSynth_VERSION_MINOR}")
  string(REGEX REPLACE "^#define +AVS_BUGFIX_VER +([0-9]+).*$" "\\1"
    AviSynth_VERSION_PATCH "${AviSynth_VERSION_PATCH}")
  string(CONCAT AviSynth_VERSION
    "${AviSynth_VERSION_MAJOR}"
    ".${AviSynth_VERSION_MINOR}"
    ".${AviSynth_VERSION_PATCH}"
  )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  AviSynth
  REQUIRED_VARS AviSynth_INCLUDE_DIR
  VERSION_VAR AviSynth_VERSION
)

if(AviSynth_FOUND)
  add_library(AviSynth INTERFACE)
  set_property(TARGET AviSynth
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${AviSynth_INCLUDE_DIR}")
endif()
