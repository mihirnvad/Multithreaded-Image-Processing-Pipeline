cmake_minimum_required(VERSION 3.16)
project(image_pipeline VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(ENABLE_TSAN "Build with ThreadSanitizer" OFF)
option(ENABLE_ASAN "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(BUILD_TESTING "Build the test executables" ON)

if(ENABLE_TSAN AND ENABLE_ASAN)
  message(FATAL_ERROR "ENABLE_TSAN and ENABLE_ASAN are mutually exclusive; use separate build directories.")
endif()

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs features2d)
message(STATUS "OpenCV ${OpenCV_VERSION}")

# ---------------------------------------------------------------------------
# Shared compile settings, applied to every target through an interface lib.
# ---------------------------------------------------------------------------
add_library(project_options INTERFACE)
target_compile_options(project_options INTERFACE -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(WARNINGS_AS_ERRORS)
  target_compile_options(project_options INTERFACE -Werror)
endif()

if(ENABLE_TSAN)
  message(STATUS "ThreadSanitizer enabled")
  target_compile_options(project_options INTERFACE -fsanitize=thread -fno-omit-frame-pointer -g -O1)
  target_link_options(project_options INTERFACE -fsanitize=thread)
endif()
if(ENABLE_ASAN)
  message(STATUS "AddressSanitizer + UndefinedBehaviorSanitizer enabled")
  target_compile_options(project_options INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer
                                                   -fno-sanitize-recover=undefined -g -O1)
  target_link_options(project_options INTERFACE -fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_library(pipeline_core STATIC
  src/ImageReader.cpp
  src/FeatureProcessor.cpp
  src/ImageWriter.cpp
  src/Pipeline.cpp
)
target_include_directories(pipeline_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# SYSTEM keeps OpenCV's own header warnings out of our -Wall -Wextra output.
target_include_directories(pipeline_core SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(pipeline_core PUBLIC ${OpenCV_LIBS} Threads::Threads PRIVATE project_options)

add_executable(image_pipeline src/main.cpp)
target_link_libraries(image_pipeline PRIVATE pipeline_core project_options)

if(BUILD_TESTING)
  enable_testing()

  # The queue test is header-only and needs no OpenCV.
  add_executable(test_queue tests/test_queue.cpp)
  target_include_directories(test_queue PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(test_queue PRIVATE Threads::Threads project_options)
  add_test(NAME test_queue COMMAND test_queue)

  add_executable(test_pipeline tests/test_pipeline.cpp)
  target_link_libraries(test_pipeline PRIVATE pipeline_core project_options)
  add_test(NAME test_pipeline COMMAND test_pipeline)

  set_tests_properties(test_queue test_pipeline PROPERTIES TIMEOUT 600)
endif()

install(TARGETS image_pipeline RUNTIME DESTINATION bin)
