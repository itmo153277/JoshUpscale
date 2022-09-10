find_package(wxWidgets REQUIRED COMPONENTS core base xrc)
include(${wxWidgets_USE_FILE})

WXWIDGETS_ADD_RESOURCES(WXRC_SRC src/wxrc_resource.xrc
  OPTIONS -e -o ${CMAKE_CURRENT_BINARY_DIR}/generated/wxrc/wxrc_resource.cc
)
add_library(JoshUpscaleGUI_WXRC STATIC ${WXRC_SRC})
target_link_libraries(JoshUpscaleGUI_WXRC PRIVATE ${wxWidgets_LIBRARIES})
file(GLOB_RECURSE SRC src/*.cc)
if (MSVC)
  list(APPEND SRC src/win_resource.rc)
endif()
add_executable(JoshUpscaleGUI WIN32 ${SRC})
target_link_libraries(JoshUpscaleGUI PRIVATE
  JoshUpscaleGUI_WXRC
  ${wxWidgets_LIBRARIES}
)
target_include_directories(JoshUpscaleGUI PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated/wxrc
)

if (ENABLE_LINT)
  if (MSVC)
    set_target_properties(JoshUpscaleGUI PROPERTIES
      VS_GLOBAL_RunCodeAnalysis true
      VS_GLOBAL_EnableMicrosoftCodeAnalysis true
      VS_GLOBAL_EnableClangTidyCodeAnalysis true
    )
  endif()
endif()
