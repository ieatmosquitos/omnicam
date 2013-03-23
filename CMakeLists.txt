project( Omnicam )

cmake_minimum_required(VERSION 2.4.6)

include_directories( ${Omnicam_SOURCE_DIR}/include )
find_package( OpenCV REQUIRED )
add_executable( Corners src/corners.cpp)
target_link_libraries( Corners ${OpenCV_LIBS})

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/misc
    ${CMAKE_BINARY_DIR}
)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
add_definitions( -DDEBUGMODE )
else()
endif()