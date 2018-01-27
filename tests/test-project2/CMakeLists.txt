## ------------------- ##
## Edit: test-project2 ##
## ------------------- ##
project(test-project2)
message(STATUS "PROCESSING ${PROJECT_NAME}")

## -------- ##
## Includes ##
## -------- ##
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)

## ---------- ##
## Executable ##
## ---------- ##
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
sdk_setup_project_bin(${PROJECT_NAME})
add_dependencies(${PROJECT_NAME} project1 project2 )
target_link_libraries(${PROJECT_NAME} project1 project2 ${LIBS} ${EXTRA_LIBS})
