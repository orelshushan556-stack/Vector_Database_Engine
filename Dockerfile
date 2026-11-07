# ==============================================================================
# Nexus Vector Database Engine
#
# Stages:
#   web-build     React / Three.js dashboard bundle
#   toolchain     Compiler, CMake, Ninja and the sources
#   engine-build  Release build of every target (library, server, tests, benchmark)
#   test          Runs the full ctest suite; the image build fails if any test fails
#   bench         Benchmark suite: scripts/run_all.py (mount datasets + results, see Stage 5)
#   asan / tsan   Sanitizer builds that run the full test suite (see Stage 6)
#   runtime       Minimal image containing nexus_server and the dashboard (default)
#
# The default target depends on the test stage, so
#   docker build -t vector-engine:latest .
# compiles, runs the complete test suite, and only then produces the runtime image.
#
# Benchmark suite (datasets are not part of the build context; see Stage 5):
#   docker build --target bench -t vector-engine:bench .
#   docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/src/results" vector-engine:bench --data-dir /data
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
# Stage 2: C++ toolchain and sources
# GoogleTest is fetched by CMake FetchContent, which requires git and network access.
# ------------------------------------------------------------------------------
FROM ubuntu:24.04 AS toolchain
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

# ------------------------------------------------------------------------------
# Stage 3: Release build of every target (-mavx2 -mfma -O3 are set in CMakeLists.txt)
# ------------------------------------------------------------------------------
FROM toolchain AS engine-build
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel

# ------------------------------------------------------------------------------
# Stage 4: test suite
# ------------------------------------------------------------------------------
FROM engine-build AS test
RUN ctest --test-dir build --output-on-failure \
    && touch /tmp/tests-passed

# ------------------------------------------------------------------------------
# Stage 5: benchmark suite (benchmarks/README.md)
# Runs scripts/run_all.py, which builds the benchmark variants (incl. hnswlib) and writes raw
# JSON to /src/results. Mount a dataset directory (scripts/download_data.py) and a results
# directory; the work directory should be a container-local or bind-mounted local disk:
#   docker build --target bench -t vector-engine:bench .
#   docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/results:/src/results" #       vector-engine:bench --machine-id <cpu-slug> --data-dir /data
# The legacy single-file SIFT-10K harness is still available:
#   docker run --rm -v "$(pwd)/data:/src/data:ro" --entrypoint /src/build/bench_sift vector-engine:bench
# Note: numbers measured in Docker Desktop run inside a VM; run_all.py records the runtime.
# ------------------------------------------------------------------------------
FROM engine-build AS bench
RUN apt-get update     && apt-get install -y --no-install-recommends python3 python3-numpy python3-h5py     && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/ scripts/
# .git is excluded from the build context, so inside the image the git commit is not recorded
# and before_after_fix (which needs git history) is reported as a gap; run it natively.
ENTRYPOINT ["python3", "scripts/run_all.py", "--build-root", "/src/build-bench"]

# ------------------------------------------------------------------------------
# Stage 6: sanitizer builds (not part of the default chain)
#   docker build --target asan .   AddressSanitizer + UndefinedBehaviorSanitizer
#   docker build --target tsan .   ThreadSanitizer
# Each stage configures with the matching ENABLE_* option, builds, and runs ctest; the
# build fails if any test fails or any sanitizer reports an error.
# ------------------------------------------------------------------------------
FROM toolchain AS asan
ENV ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1:halt_on_error=1 \
    UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
RUN cmake -S . -B build_asan -G Ninja -DENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug \
    && cmake --build build_asan --parallel \
    && ctest --test-dir build_asan --output-on-failure

FROM toolchain AS tsan
ENV TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1
# ThreadSanitizer cannot start on kernels whose address-space randomization is wider than
# its fixed shadow-memory layout ("unexpected memory mapping"), so this stage only builds.
# Run the suite with ASLR disabled, which needs the personality() syscall:
#   docker build --target tsan -t vector-engine:tsan .
#   docker run --rm --security-opt seccomp=unconfined vector-engine:tsan
# Test discovery is deferred to ctest time for the same reason.
RUN cmake -S . -B build_tsan -G Ninja -DENABLE_TSAN=ON -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST \
    && cmake --build build_tsan --parallel
CMD ["setarch", "x86_64", "-R", "ctest", "--test-dir", "build_tsan", "--output-on-failure"]

# ------------------------------------------------------------------------------
# Stage 7: minimal runtime
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
