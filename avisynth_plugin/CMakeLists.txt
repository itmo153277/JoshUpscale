find_package(AviSynth REQUIRED)

file(GLOB_RECURSE SRC src/*.cc)
add_library(JoshUpscaleAvisynth SHARED ${SRC})
target_link_libraries(JoshUpscaleAvisynth PRIVATE
  AviSynth
  JoshUpscale
)

if (ENABLE_LINT)
  if (MSVC)
    set_target_properties(JoshUpscaleAvisynth PROPERTIES
      VS_GLOBAL_RunCodeAnalysis true
      VS_GLOBAL_EnableMicrosoftCodeAnalysis true
      VS_GLOBAL_EnableClangTidyCodeAnalysis true
    )
  endif()
endif()
