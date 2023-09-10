find_package(FFMPEG REQUIRED)
find_package(LibObs REQUIRED)

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
target_link_directories(obs-joshupscale PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(obs-joshupscale PRIVATE
  JoshUpscale
  libobs
  ${FFMPEG_LIBRARIES}
)
