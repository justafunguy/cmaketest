cmake_minimum_required(VERSION 3.16)
#项目的名称，可以和所在文件夹名称不同
project(cmaketest)

add_library(add_shared SHARED add.cpp)
set_target_properties(add_shared PROPERTIES OUTPUT_NAME "add")
#set_target_properties(add_shared PROPERTIES CLEAN_DIRECT_OUTPUT 1)
add_library(add STATIC add.cpp)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/)
#set_target_properties(add PROPERTIES CLEAN_DIRECT_OUTPUT 1)
#set_target_properties(add PROPERTIES VERSION 1.1 SOVERSION 1.0)
#install(TARGETS add_shared add LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
#install(FILES add.h DESTINATION include/hello/)