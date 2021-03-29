add_lab("INVERSE_MATRIX_test")
add_lab_solution("INVERSE_MATRIX_test", ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("INVERSE_MATRIX_test" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
