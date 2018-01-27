## -------------- ##
## Edit: project1 ##
## -------------- ##
project(project1)
message(STATUS "PROCESSING ${PROJECT_NAME}")

## -------- ##
## Includes ##
## -------- ##
sdk_list_header_files(HEADER_FILES)
sdk_list_source_files(SOURCE_FILES)

## ----------------------- ##
## Interface Library Build ##
## ----------------------- ##
add_library(${PROJECT_NAME} INTERFACE)
target_compile_definitions(${PROJECT_NAME} INTERFACE ${DEFINITIONS})
target_include_directories(${PROJECT_NAME} INTERFACE ${HEADER_FILES})

## -------------------- ##
## Static Library Build ##
## -------------------- ##
#add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
#target_compile_definitions(${PROJECT_NAME} PUBLIC ${DEFINITIONS})
#target_link_libraries(${PROJECT_NAME})
#sdk_setup_project_lib(${PROJECT_NAME})
