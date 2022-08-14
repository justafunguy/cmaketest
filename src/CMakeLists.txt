#添加头文件的搜索路径
include_directories(${PROJECT_SOURCE_DIR}/header )
link_directories(${PROJECT_SOURCE_DIR}/lib)
add_link_options(-static)
#将源文件列表写在变量SrcFiles中
aux_source_directory(${PROJECT_SOURCE_DIR}/src SrcFiles)
#message(SEND_ERROR "ABC")

add_executable(a.out ${SrcFiles})
target_link_libraries(a.out add)
