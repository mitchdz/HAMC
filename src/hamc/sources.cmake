add_lab("hamc")
add_lab_solution("hamc" ${CMAKE_CURRENT_LIST_DIR}/main.cu)

#add_lab("hamc")

#file(GLOB HAMC_SRCS
#    ${CMAKE_CURRENT_LIST_DIR}*.cu
#)
#message( "${HAMC_SRCS}" )

#cuda_include_directories(${CMAKE_CURRENT_LIST_DIR})
#include_directories(${CMAKE_CURRENT_LIST_DIR})
#cuda_add_executable(hamc ${CMAKE_CURRENT_LIST_DIR}/hamc.cu ${HAMC_SRCS})
#target_link_libraries(hamc ${WBLIB} ${LINK_LIBRARIES})

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -rdc=true --device-c")
#add_lab_helper(hamc ${CMAKE_CURRENT_LIST_DIR}/hamc.cu ${CU_O})

#file(GLOB CUDA_FILES ${CMAKE_CURRENT_LIST_DIR}/kernels/*)
#add_library(hamc_lib ${CUDA_FILES})
#
#include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
#add_executable(hamc)
#target_sources(hamc ${CUDA_FILES})


#target_link_libraries("hamc" "${WBLIB}" "${LINK_LIBRARIES} hamc_lib")

#CUDA_INCLUDE_DIRS(hamc ${CMAKE_CURRENT_LIST_DIR})
#add_lab_helper(hamc ${CMAKE_CURRENT_LIST_DIR}/hamc.cu
#    ${CMAKE_CURRENT_LIST_DIR}/encrypt.cu
#    ${CMAKE_CURRENT_LIST_DIR}/decrypt.cu
#    ${CMAKE_CURRENT_LIST_DIR}/hamc_common.cu
#    ${CMAKE_CURRENT_LIST_DIR}/hamc.cu
#    ${CMAKE_CURRENT_LIST_DIR}/HAMC_decrypt.cu
#    ${CMAKE_CURRENT_LIST_DIR}/HAMC_encrypt.cu
#    ${CMAKE_CURRENT_LIST_DIR}/HAMC_key_gen.cu
#    ${CMAKE_CURRENT_LIST_DIR}/InverseMatrix.cu
#    ${CMAKE_CURRENT_LIST_DIR}/keygen.cu
#    ${CMAKE_CURRENT_LIST_DIR}/MatrixAdd.cu
#    ${CMAKE_CURRENT_LIST_DIR}/matrix.cu
#    ${CMAKE_CURRENT_LIST_DIR}/mceliece.cu
#    ${CMAKE_CURRENT_LIST_DIR}/MultiplyMatrix.cu
#    ${CMAKE_CURRENT_LIST_DIR}/qc_mdpc.cu
#    ${CMAKE_CURRENT_LIST_DIR}/RREFMatrix.cu
#    ${CMAKE_CURRENT_LIST_DIR}/TransposeMatrix.cu
#    ${CMAKE_CURRENT_LIST_DIR}/qc_mdpc.h
#    ${CMAKE_CURRENT_LIST_DIR}/hamc_common.h
#    ${CMAKE_CURRENT_LIST_DIR}/mceliece.h
#    ${CMAKE_CURRENT_LIST_DIR}/matrix.h
#    ${CMAKE_CURRENT_LIST_DIR}/keygen.h
#    ${CMAKE_CURRENT_LIST_DIR}/MultiplyMatrix.h
#    )


#include_directories(${CMAKE_CURRENT_LIST_DIR})
#
#file( GLOB HAMC_SRCS
#    ${CMAKE_CURRENT_LIST_DIR}/*
#)
#
#add_executable(hamc ${CMAKE_CURRENT_LIST_DIR}/main.cu ${HAMC_SRCS})
#target_link_libraries(hamc ${WBLIB} ${LINK_LIBRARIES})


