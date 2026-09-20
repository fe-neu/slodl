cmake_minimum_required(VERSION 3.16...3.30)
project(slodl LANGUAGES CXX)


add_library(slodl_core STATIC
    src/cpp/slodl/tensor/tensor.cpp
    src/cpp/slodl/tensor/tensor_storage.cpp
    )
target_include_directories(slodl_core PUBLIC src/cpp)
target_compile_features(slodl_core PUBLIC cxx_std_17)
set_target_properties(slodl_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# --- Python extension --------------------------------------------------------
# On by default (scikit-build-core drives the wheel build this way). Turn it
# off for a tests-only dev build that has no pybind11 in scope:
#   cmake -S . -B build-test -DSLODL_BUILD_TESTS=ON -DSLODL_BUILD_PYTHON=OFF
option(SLODL_BUILD_PYTHON "Build the pybind11 extension module" ON)
if(SLODL_BUILD_PYTHON)
    # Let pybind11 locate the interpreter scikit-build-core selected.
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    # The compiled module. Only the binding glue lives here; everything it
    # wraps comes from slodl_core. Add each new binding .cpp below.
    pybind11_add_module(_core
        src/cpp/slodl/bindings/_core.cpp
        src/cpp/slodl/bindings/conversions.cpp
        src/cpp/slodl/bindings/tensor/tensor.cpp
    )
    target_link_libraries(_core PRIVATE slodl_core)

    install(TARGETS _core DESTINATION slodl)
endif()

# --- Native C++ tests --------------------------------------------------------
# Off by default so `pip install` / wheel builds never need the test
# framework. Enable a dev build with:
#   cmake -S . -B build-test -DSLODL_BUILD_TESTS=ON
#   cmake --build build-test
#   ctest --test-dir build-test --output-on-failure
option(SLODL_BUILD_TESTS "Build the C++ unit tests" OFF)
if(SLODL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
