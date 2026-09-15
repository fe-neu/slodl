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
# put testfiles here
)
target_link_libraries(slodl_cpp_tests
    PRIVATE
        slodl_core
        Catch2::Catch2WithMain
)

# Register each TEST_CASE as an individual CTest entry.
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(slodl_cpp_tests)
