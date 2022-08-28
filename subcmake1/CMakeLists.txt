project(subcmake1)
add_library(${PROJECT_NAME} STATIC add.cpp )
message("in subcmake1: project_name=${PROJECT_NAME}")