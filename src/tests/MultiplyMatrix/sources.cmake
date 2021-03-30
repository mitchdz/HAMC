add_lab("Multiply_test")
add_lab_solution("Multiply_test" ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("Multiply_test" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
