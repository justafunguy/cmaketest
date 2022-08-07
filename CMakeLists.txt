cmake_minimum_required(VERSION 3.16)
#项目的名称，可以和所在文件夹名称不同
project(mycmaketest)
#添加头文件的搜索路径
include_directories(${PROJECT_SOURCE_DIR}/header)

#将源文件列表写在变量SrcFiles中
aux_source_directory(./src SrcFiles)

#设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
add_compile_options(-ggdb3)
add_link_options(-static)
add_executable(a.out ${SrcFiles})