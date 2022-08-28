cmake_minimum_required(VERSION 3.10)
project(CMakeLists.txt&&Makefile)
include_directories(${CMAKE_SOURCE_DIR}/header)
include(test.cmake)
set(abc "ABC")
fn1(${abc})

if(abc STREQUAL "def")
    message("abc==def!")
endif()
message("abc=${abc}")
configure_file(main.cpp ${CMAKE_SOURCE_DIR}/main.cpp)
add_executable(a.out main.cpp test.cpp)


