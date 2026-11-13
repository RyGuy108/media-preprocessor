cmake_minimum_required(VERSION 3.20)
project(mhprep LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MHPREP_BUILD_TESTS "Build GoogleTest suite" ON)
option(MHPREP_SANITIZE "Build with ASan + UBSan" OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)
pkg_check_modules(PQ REQUIRED IMPORTED_TARGET libpq)
# cpp-httplib: Debian/Ubuntu ship it as a compiled library (cpp-httplib.pc); elsewhere
# it is header-only.
pkg_check_modules(HTTPLIB IMPORTED_TARGET cpp-httplib)
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)

add_compile_options(-Wall -Wextra -Wpedantic)
if(MHPREP_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(mhprep_core STATIC
  src/phash.cpp
  src/hash_index.cpp
  src/scene.cpp
  src/faces.cpp
  src/video_source.cpp
  src/pipeline.cpp
  src/ipc.cpp
  src/supervisor.cpp
  src/worker.cpp
  src/metrics.cpp
  src/job_store.cpp
)
target_include_directories(mhprep_core PUBLIC include)
target_link_libraries(mhprep_core PUBLIC
  ${OpenCV_LIBS} nlohmann_json::nlohmann_json PkgConfig::LIBAV PkgConfig::PQ Threads::Threads)

add_executable(mhprep src/main.cpp src/server.cpp)
target_include_directories(mhprep PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(mhprep PRIVATE mhprep_core)
if(HTTPLIB_FOUND)
  target_link_libraries(mhprep PRIVATE PkgConfig::HTTPLIB)
endif()

if(MHPREP_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(mhprep_tests
    tests/test_phash.cpp
    tests/test_hash_index.cpp
    tests/test_scene.cpp
    tests/test_ipc.cpp
    tests/test_pipeline.cpp
    tests/test_supervisor.cpp
    tests/test_job_store.cpp
  )
  target_link_libraries(mhprep_tests PRIVATE mhprep_core GTest::gtest_main)
  target_compile_definitions(mhprep_tests PRIVATE
    MHPREP_BIN="$<TARGET_FILE:mhprep>"
    MHPREP_MODELS="${CMAKE_SOURCE_DIR}/models")
  add_dependencies(mhprep_tests mhprep)
  include(GoogleTest)
  gtest_discover_tests(mhprep_tests DISCOVERY_TIMEOUT 30)
endif()
