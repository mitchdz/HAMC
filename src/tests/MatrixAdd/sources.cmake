add_lab("ADD_MATRIX_test")
add_lab_solution("ADD_MATRIX_test", ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("ADD_MATRIX_test" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)