# syntax=docker/dockerfile:1
#
# One-command verification:
#   docker build -t image-pipeline .
#   docker run --rm -v "$PWD/results:/app/results" -v "$PWD/assets:/app/assets" image-pipeline
#
# The build stage compiles the pipeline and runs the test suite (including a
# ThreadSanitizer build of the queue stress test), so a successful
# `docker build` is itself proof that the code is correct and race-free.

# ---------------------------------------------------------------------------
# Stage 1: build + test
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential clang libclang-rt-dev cmake ninja-build libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include ./include
COPY src ./src
COPY tests ./tests

# Release build with warnings as errors, then run the unit + integration tests.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWARNINGS_AS_ERRORS=ON \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure

# ThreadSanitizer build: the 32-thread queue stress test and the end-to-end
# pipeline test must both finish with zero reports.
RUN CXX=clang++ cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON -DBUILD_TESTING=ON \
    && cmake --build build-tsan --target test_queue test_pipeline \
    && export TSAN_OPTIONS="halt_on_error=1 second_deadlock_stack=1 suppressions=/src/tests/tsan.supp" \
    && ./build-tsan/test_queue && ./build-tsan/test_pipeline

# ---------------------------------------------------------------------------
# Stage 2: runtime (binary + Python tooling, no compilers)
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
# python3-opencv pulls in the OpenCV shared libraries the binary links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-numpy python3-opencv python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/image_pipeline /usr/local/bin/image_pipeline
COPY scripts ./scripts
RUN chmod +x scripts/*.sh

# Generate the dataset, benchmark every thread count, render the charts.
CMD ["./scripts/run_all.sh"]
