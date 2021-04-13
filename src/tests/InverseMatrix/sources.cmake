add_lab("inverse_test")
add_lab_solution("inverse_test" ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("inverse_gen_matrix" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
