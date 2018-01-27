## ------------------- ##
## Edit: project3-CUDA ##
## ------------------- ##
project(project3-CUDA)
message(STATUS "PROCESSING ${PROJECT_NAME}")

## -------- ##
## Includes ##
## -------- ##
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)

## -------------------- ##
## Static Library Build ##
## -------------------- ##
cuda_add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties( ${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON )
target_compile_definitions(${PROJECT_NAME} PUBLIC ${DEFINITIONS})
target_link_libraries(${PROJECT_NAME})
sdk_setup_project_lib(${PROJECT_NAME})
