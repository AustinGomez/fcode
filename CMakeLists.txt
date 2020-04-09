project(fcode)
set(CMAKE_CXX_STANDARD 14)
set(dir ${CMAKE_CURRENT_SOURCE_DIR}/.)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${dir})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main videoChunked.cpp)

# linking
target_link_libraries(main ${OpenCV_LIBS})