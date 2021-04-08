add_lab("MatrixAdd_test")
add_lab_solution("MatrixAdd_test", ${CMAKE_CURRENT_LIST_DIR}/main.cu)
add_generator("MatrixAdd" ${CMAKE_CURRENT_LIST_DIR}/dataset_generator.cpp)
