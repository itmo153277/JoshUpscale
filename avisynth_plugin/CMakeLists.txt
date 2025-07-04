find_package(AviSynth REQUIRED)

file(GLOB_RECURSE SRC src/*.cc)
add_library(JoshUpscaleAvisynth SHARED ${SRC})
target_link_libraries(JoshUpscaleAvisynth PRIVATE
  AviSynth
  JoshUpscale
)

install(
  TARGETS JoshUpscaleAvisynth
  RUNTIME
    COMPONENT Runtime
  LIBRARY
    COMPONENT Runtime
    NAMELINK_SKIP
)
