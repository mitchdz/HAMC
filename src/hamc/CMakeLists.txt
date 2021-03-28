cmake_minimum_required (VERSION 2.6)
project (hamc)
add_definitions(-Wall -g)

file( GLOB SRCS *.cu *.cpp *.h )

add_executable(hamc ${SRCS})
