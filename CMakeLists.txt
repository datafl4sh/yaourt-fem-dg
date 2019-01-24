cmake_minimum_required (VERSION 3.5)
project (dg2d)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Additional modules path for cmake
set (CMAKE_MODULE_PATH
     ${CMAKE_MODULE_PATH}
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_FLAGS_DEBUG "-g -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -fpermissive -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASEASSERT "-O3 -g -fpermissive")

find_package(SILO REQUIRED)
if (SILO_FOUND)
    add_definitions(-DWITH_SILO)
    include_directories("${SILO_INCLUDE_DIRS}")
    set(LINK_LIBS ${LINK_LIBS} ${SILO_LIBRARIES})
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}")

include_directories(blaze)
add_subdirectory(blaze)

add_executable(dg2d dg2d.cpp)
target_link_libraries(dg2d ${LINK_LIBS})

