add_lab("hamc")
file( GLOB HAMC_SRCS ${CMAKE_CURRENT_LIST_DIR}/*.cu ${CMAKE_CURRENT_LIST_DIR}/*.h ${CMAKE_CURRENT_LIST_DIR}/*.c )
add_lab_solution("hamc" ${HAMC_SRCS})
