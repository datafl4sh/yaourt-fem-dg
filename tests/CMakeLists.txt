add_executable(quadratures quadratures.cpp)
target_link_libraries(quadratures ${LINK_LIBS})

add_executable(refelem refelem.cpp)
target_link_libraries(refelem ${LINK_LIBS})

#add_executable(maxwell_refelem maxwell_refelem.cpp)
#target_link_libraries(maxwell_refelem ${LINK_LIBS})
