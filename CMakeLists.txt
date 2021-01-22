cmake_minimum_required (VERSION 3.5)
project (dg2d)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Additional modules path for cmake
set (CMAKE_MODULE_PATH
     ${CMAKE_MODULE_PATH}
     ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wconversion -Wno-sign-conversion -Wno-unused-variable -Wno-shorten-64-to-32")

#set(CMAKE_CXX_FLAGS_DEBUG "-g -fpermissive -fsanitize=address")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fpermissive")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fpermissive -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASEASSERT "-O3 -g -fpermissive")

find_package(SILO)
if (SILO_FOUND)
    add_definitions(-DWITH_SILO)
    include_directories("${SILO_INCLUDE_DIRS}")
    set(LINK_LIBS ${LINK_LIBS} ${SILO_LIBRARIES})
endif()

find_package(LAPACK REQUIRED)
if(LAPACK_FOUND)
    set(LINK_LIBS ${LINK_LIBS} ${LAPACK_LIBRARIES})
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}")

include_directories(blaze)
add_subdirectory(blaze)

add_subdirectory(tests)

add_executable(dg2d_advection dg2d_advection.cpp)
target_link_libraries(dg2d_advection ${LINK_LIBS})

add_executable(dg2d_diffusion dg2d_diffusion.cpp)
target_link_libraries(dg2d_diffusion ${LINK_LIBS})

#add_executable(maxwell maxwell_driver.cpp)
#target_link_libraries(maxwell ${LINK_LIBS})

#add_executable(continuous_fem continuous_fem.cpp)
#target_link_libraries(continuous_fem ${LINK_LIBS})

#add_executable(test_quad test_quad.cpp)
#target_link_libraries(test_quad ${LINK_LIBS})

add_executable(ex1_geometry ex1_geometry.cpp)
target_link_libraries(ex1_geometry ${LINK_LIBS})

add_executable(ex2_quadrature ex2_quadrature.cpp)
target_link_libraries(ex2_quadrature ${LINK_LIBS})

add_executable(ex3_projection ex3_projection.cpp)
target_link_libraries(ex3_projection ${LINK_LIBS})

