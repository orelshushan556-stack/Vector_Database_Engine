# ==============================================================================
# Nexus Vector Database Engine
#
# Stages:
#   web-build     React / Three.js dashboard bundle
#   engine-build  Release build of every target (library, server, tests, benchmark)
#   test          Runs the full ctest suite; the image build fails if any test fails
#   bench         SIFT benchmark harness (run with the dataset mounted, see below)
#   runtime       Minimal image containing nexus_server and the dashboard (default)
#
# The default target depends on the test stage, so
#   docker build -t vector-engine:latest .
# compiles, runs the complete test suite, and only then produces the runtime image.
#
# Benchmark (the SIFT-small files are not part of the build context):
#   docker build --target bench -t vector-engine:bench .
#   docker run --rm -v "$(pwd)/data:/src/data:ro" vector-engine:bench
#
# The engine requires an AVX2 and FMA capable CPU (compiled with -mavx2 -mfma).
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: web frontend build
# vite is configured with outDir '../public', so the bundle lands in /app/public
# ------------------------------------------------------------------------------
FROM node:20-alpine AS web-build
WORKDIR /app/web

COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: C++ engine build (Release; -mavx2 -mfma -O3 are set in CMakeLists.txt)
# GoogleTest is fetched by CMake FetchContent, which requires git and network access.
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS engine-build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        g++ cmake ninja-build git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY examples/ examples/
COPY benchmarks/ benchmarks/
COPY tests/ tests/

RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel

# ------------------------------------------------------------------------------
# Stage 3: test suite
# ------------------------------------------------------------------------------
FROM engine-build AS test
RUN ctest --test-dir build --output-on-failure \
    && touch /tmp/tests-passed

# ------------------------------------------------------------------------------
# Stage 4: SIFT benchmark harness
# Expects data/siftsmall/{siftsmall_base.fvecs, siftsmall_query.fvecs,
# siftsmall_groundtruth.ivecs} under /src/data (mount it at run time).
# ------------------------------------------------------------------------------
FROM engine-build AS bench
WORKDIR /src
ENTRYPOINT ["/src/build/bench_sift"]

# ------------------------------------------------------------------------------
# Stage 5: minimal runtime
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=engine-build /src/build/nexus_server ./nexus_server
COPY --from=web-build /app/public ./public

# Ties this stage to the test stage so a default build cannot succeed with failing tests.
COPY --from=test /tmp/tests-passed /etc/nexus-tests-passed

# The server opens "nexus.wal" relative to its working directory. Point it at the
# persistent data directory so the WAL survives container restarts.
RUN mkdir -p /app/data && ln -s /app/data/nexus.wal /app/nexus.wal

VOLUME ["/app/data"]
EXPOSE 8080

ENTRYPOINT ["./nexus_server"]
