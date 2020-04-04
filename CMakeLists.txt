cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(PistachePredictor)

set(CMAKE_CXX_STANDARD 14)

find_package(Pistache REQUIRED)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/onnxruntime-linux-x64-1.2.0/lib")

add_executable(${PROJECT_NAME} src/server.cc)

include_directories("/usr/include/rdkit")
include_directories("/onnxruntime-linux-x64-1.2.0/include")
target_link_libraries(${PROJECT_NAME} pistache pthread RDKitSmilesParse RDKitGraphMol RDKitFingerprints ${ONNXRUNTIME_LIBRARY})
