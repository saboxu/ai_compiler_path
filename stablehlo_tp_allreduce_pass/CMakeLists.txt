cmake_minimum_required(VERSION 3.20)
project(stablehlo_tp_allreduce_pass LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to the freshly-built LLVM/MLIR tree under third_party/stablehlo.
get_filename_component(PROJ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DEFAULT_LLVM_CMAKE_DIR
  "${PROJ_ROOT}/third_party/stablehlo/llvm-build/lib/cmake/llvm")
set(DEFAULT_MLIR_CMAKE_DIR
  "${PROJ_ROOT}/third_party/stablehlo/llvm-build/lib/cmake/mlir")

if(NOT DEFINED LLVM_DIR)
  set(LLVM_DIR "${DEFAULT_LLVM_CMAKE_DIR}" CACHE PATH "Path to LLVMConfig.cmake")
endif()
if(NOT DEFINED MLIR_DIR)
  set(MLIR_DIR "${DEFAULT_MLIR_CMAKE_DIR}" CACHE PATH "Path to MLIRConfig.cmake")
endif()

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJ_ROOT}/third_party/stablehlo)
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})

set(STABLEHLO_LIB_DIR "${PROJ_ROOT}/third_party/stablehlo/build/lib")

set(STABLEHLO_DRIVER_LIBS
  ${STABLEHLO_LIB_DIR}/libStablehloRegister.a
  ${STABLEHLO_LIB_DIR}/libStablehloOps.a
  ${STABLEHLO_LIB_DIR}/libChloOps.a
  ${STABLEHLO_LIB_DIR}/libVhloOps.a
  ${STABLEHLO_LIB_DIR}/libVhloTypes.a
  ${STABLEHLO_LIB_DIR}/libStablehloBase.a
  ${STABLEHLO_LIB_DIR}/libStablehloAssemblyFormat.a
  ${STABLEHLO_LIB_DIR}/libStablehloTypeInference.a
  ${STABLEHLO_LIB_DIR}/libStablehloBroadcastLowering.a
  ${STABLEHLO_LIB_DIR}/libStablehloBroadcastUtils.a
  ${STABLEHLO_LIB_DIR}/libStablehloReplicaGroupUtils.a
  ${STABLEHLO_LIB_DIR}/libStablehloPassUtils.a
  ${STABLEHLO_LIB_DIR}/libAttrTypeBuilderUtil.a
  ${STABLEHLO_LIB_DIR}/libVersion.a
)

add_llvm_library(stablehlo_tp_allreduce_pass
  MODULE
  BUILDTREE_ONLY
  src/StablehloTPAllReducePass.cpp

  PLUGIN_TOOL
  mlir-opt
)

add_llvm_executable(stablehlo-tp-opt
  src/StablehloTPOptMain.cpp
  src/StablehloTPAllReducePass.cpp
)
llvm_update_compile_flags(stablehlo-tp-opt)
target_link_libraries(stablehlo-tp-opt PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSupport
  ${STABLEHLO_DRIVER_LIBS}
)

