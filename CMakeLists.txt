cmake_minimum_required(VERSION 3.16)
#项目的名称，可以和所在文件夹名称不同
project(cmaketest)
#设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_link_options(-static)

add_subdirectory(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/src/build )

#INSTALL(FILES README COPYRIGHT DESTINATION share/doc/cmake/)
#INSTALL(PROGRAMS runhello.sh DESTINATION bin)
#INSTALL(DIRECTORY doc/ DESTINATION share/doc/cmake/ )
#INSTALL(TARGETS /home/wg/cppjob/worker/cmaketest/bin/a.out DESTINATION bin)
