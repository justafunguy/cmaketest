project(subcmake2)
add_library(${PROJECT_NAME} STATIC sub.cpp )
message("in subcmake2: project_name=${PROJECT_NAME}")