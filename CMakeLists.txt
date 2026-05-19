cmake_minimum_required(VERSION 3.16)
project(FacialRec VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find OpenCV ───────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect face)

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/detector.cpp
    src/recognizer.cpp
    src/renderer.cpp
    src/utils.cpp
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(facialrec ${SOURCES})

target_include_directories(facialrec
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(facialrec
    PRIVATE
        ${OpenCV_LIBS}
)

# ── Compiler warnings ─────────────────────────────────────────────────────────
target_compile_options(facialrec PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)
