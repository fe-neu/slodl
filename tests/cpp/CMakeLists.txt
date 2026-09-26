# Catch2 v3, fetched at configure time. Swap GIT_TAG for a newer release as
# desired; the download is cached in the build tree after the first configure.
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# One aggregate test binary. Mirror src/cpp/slodl/<area>/<file>.cpp with a
# tests/cpp/<area>/test_<file>.cpp entry here as the suite grows.
add_executable(slodl_cpp_tests
    tensor/test_tensor.cpp
    tensor/test_tensor_storage.cpp
    tensor/test_ops.cpp
    autograd/test_autograd.cpp
    autograd/test_grad_mode.cpp
    autograd/test_ops.cpp
)
target_link_libraries(slodl_cpp_tests
    PRIVATE
        slodl_core
        Catch2::Catch2WithMain
)

# Register each TEST_CASE as an individual CTest entry.
# One discovery pass per source area, so every test name gets an "<area>/"
# prefix. With "cmake.ctest.testSuiteDelimiter": "/" the VS Code Testing view
# then groups them into a tree instead of one flat list. Each area has one tag
# shared by all its test files ([tensor] here), so a new file in an existing
# area needs no change below; a new area adds its own line.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(slodl_cpp_tests TEST_PREFIX "tensor/" TEST_SPEC "[tensor]")
catch_discover_tests(slodl_cpp_tests TEST_PREFIX "autograd/" TEST_SPEC "[autograd]")
