cmake_minimum_required(VERSION 3.6)
project(PistachePredictor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PkgConfig)
pkg_check_modules(Pistache REQUIRED IMPORTED_TARGET libpistache)

find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/onnxruntime-linux-x64-1.12.1/lib")

include_directories("/usr/include/rdkit" "/onnxruntime-linux-x64-1.12.1/include")

set(RDKit_LIBS RDKitSmilesParse RDKitGraphMol RDKitFingerprints)

add_executable(${PROJECT_NAME} src/server.cc)
target_link_libraries(${PROJECT_NAME} PkgConfig::Pistache pthread ${RDKit_LIBS} ${ONNXRUNTIME_LIBRARY})
