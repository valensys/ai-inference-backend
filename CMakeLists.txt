if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

cmake_minimum_required(VERSION 2.6)

project(Analytics)

#add_definitions(-std=c++20)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -Wall -lpthread -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#onnxruntime
include_directories(/usr/local/onnxruntime/include)
link_directories(/usr/local/onnxruntime/lib)

#byte track
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/ByteTrack)

#http server
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/HttpServer)

#siz classifier
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/AITools)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/OtherTools)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Poco REQUIRED COMPONENTS Foundation Net)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${Poco_INCLUDE_DIRS}
                    ${Eigen3_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SRC_CODE "src/*.cpp")
file(GLOB BYTE_TRACK_CODE "src/ByteTrack/*.cpp")
file(GLOB HTTP_SERVER_CODE "src/HttpServer/*.cpp")
file(GLOB HTTP_SERVER_HANDLERS_CODE "src/HttpServer/handlers/*.cpp")
file(GLOB AI_TOOLS_CODE "src/AITools/*.cpp")
file(GLOB RESTREAMING_CODE "src/Restreaming/*.cpp")
file(GLOB OTHER_TOOLS_CODE "src/OtherTools/*.cpp")


add_executable(${PROJECT_NAME} 
                ${SRC_CODE} 
                ${BYTE_TRACK_CODE} 
                ${HTTP_SERVER_CODE} 
                ${HTTP_SERVER_HANDLERS_CODE} 
                ${AI_TOOLS_CODE} 
                ${RESTREAMING_CODE} 
                ${OTHER_TOOLS_CODE})

target_link_libraries(Analytics 
                        ${OpenCV_LIBS}
                        PocoUtil
                        PocoNet
                        PocoFoundation
                        onnxruntime)

add_definitions(-O2 -pthread)
