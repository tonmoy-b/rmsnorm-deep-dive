cmake_minimum_required(VERSION 3.26)
project(rmsnorm_ext LANGUAGES CXX CUDA)

# --- Standards ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- CUDA arch (use TORCH_CUDA_ARCH_LIST; torch overrides CMAKE_CUDA_ARCHITECTURES) -
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST "8.6")
endif()

# --- Locate Python -----------------------------------------------------------
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# --- Locate Torch (FORCE conda env's torch, ignore stray libtorch installs) --
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import torch, os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
message(STATUS "Using torch from: ${TORCH_INSTALL_PREFIX}")

# Override anything else by putting this first AND setting Torch_DIR explicitly.
set(Torch_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/Torch" CACHE PATH "" FORCE)
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake")
find_package(Torch REQUIRED)

# --- Locate pybind11 from the active env -------------------------------------
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# --- The extension target ----------------------------------------------------
pybind11_add_module(_C MODULE
  csrc/rmsnorm.cpp
  csrc/rmsnorm_cuda.cu
)

#target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} Python::Module)
# torch_python.lib provides the pybind11 type_caster<at::Tensor> specialization.
# It's not in ${TORCH_LIBRARIES} by default; we have to add it explicitly.
find_library(TORCH_PYTHON_LIBRARY torch_python
             PATHS "${TORCH_INSTALL_PREFIX}/lib"
             NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_C PRIVATE
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  Python::Module
)
target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(_C PRIVATE ${TORCH_CXX_FLAGS})

target_compile_options(_C PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:/O2 /Zc:__cplusplus /Zc:preprocessor /permissive- /bigobj /wd4624 /wd4267>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math --expt-relaxed-constexpr -Xcompiler=/bigobj -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus>
)

set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/rmsnorm"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/rmsnorm"
  LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/rmsnorm"
  RUNTIME_OUTPUT_DIRECTORY         "${CMAKE_SOURCE_DIR}/rmsnorm"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/rmsnorm"
  RUNTIME_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_SOURCE_DIR}/rmsnorm"
)